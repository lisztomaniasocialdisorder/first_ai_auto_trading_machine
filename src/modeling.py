from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from .config import Settings


def _asymmetric_lev_weights(
    lev_train: np.ndarray,
    penalty_factor: float = 3.0,
) -> np.ndarray:
    """
    ??蝔望見?祆??????
    ?∪極?摩嚗??鈭斗?鋆∴?擃?獢踹????葫隤方炊隞???雿?獢踹??之??    ?迨霈芋??撠見?砌葉??瑽▼鞈?暺??箸憭找誨?嫘?
    penalty_factor嚗?瑽▼璅????撠?撠?瑽▼璅?嚗?.0~5.0 ?箔蔔嚗?    """
    lev_min, lev_max = lev_train.min(), lev_train.max()
    if lev_max <= lev_min:
        return np.ones(len(lev_train), dtype=np.float64)
    # 甇訾?? [0, 1]??甈⊥?脩?嚗?獢輯?擃蝵啗?憭改????扳憭折?瑽▼???
    norm = ((lev_train - lev_min) / (lev_max - lev_min)) ** 2
    weights = 1.0 + (penalty_factor - 1.0) * norm
    return (weights / weights.mean()).clip(0.3, penalty_factor * 1.5)


def _pinball_sample_weights(
    lev_train: np.ndarray,
    lev_pred_warm: np.ndarray,
    tau: float = 0.35,
) -> np.ndarray:
    """
    Pinball/Quantile ?仃?見?祆???隡潦?
    Pinball loss ?砍?嚗?      L(y, y_hat) = (y - y_hat) * tau              ??y >= y_hat  (?葫??)
      L(y, y_hat) = (y_hat - y) * (1 - tau)        ??y < y_hat   (?葫?? ???脩蔑?游?)

    tau=0.35嚗?璅∪???敹?隡唳?獢踴?撖嫘?隡唳?獢踴??脩蔑?箔?隡啁? (1-0.35)/0.35 ??1.86 ??    """
    residual = lev_train - lev_pred_warm
    weights = np.where(
        residual >= 0,
        tau,
        1 - tau,
    )
    return (weights / weights.mean()).clip(0.2, 4.0)


@dataclass
class TrainedModels:
    clf: Any
    lev_reg: Any
    feature_cols: list[str]
    backend: str = "sklearn_rf"
    backend_meta: dict | None = None


class EnsembleLevReg:
    """
    Top-level ensemble regressor so joblib/pickle can serialize it.
    Combines a warm-up RF and a quantile GB regressor.
    """

    def __init__(self, rf: Any, gb: Any, rf_w: float = 0.40) -> None:
        self.rf = rf
        self.gb = gb
        self.rf_w = rf_w

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rf_p = self.rf.predict(X)
        gb_p = self.gb.predict(X)
        blended = self.rf_w * rf_p + (1.0 - self.rf_w) * gb_p
        return np.clip(blended * 0.95, 1.0, 1e6)


class _TorchSignalWrapper:
    def __init__(self, model: Any, device: Any, mean: np.ndarray, scale: np.ndarray, class_values: np.ndarray, torch_mod: Any) -> None:
        self.model = model
        self.device = device
        self.mean = mean.astype(np.float32)
        self.scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
        self.classes_ = class_values.astype(int)
        self._torch = torch_mod

    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        arr = x.to_numpy(dtype=np.float32, copy=False)
        return (arr - self.mean) / self.scale

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with self._torch.no_grad():
            xt = self._torch.from_numpy(self._transform(x)).to(self.device)
            logits = self.model(xt)
            p = self._torch.softmax(logits, dim=1).detach().cpu().numpy()
        return p

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(x)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


class _TorchLevWrapper:
    def __init__(self, model: Any, device: Any, mean: np.ndarray, scale: np.ndarray, torch_mod: Any) -> None:
        self.model = model
        self.device = device
        self.mean = mean.astype(np.float32)
        self.scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
        self._torch = torch_mod

    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        arr = x.to_numpy(dtype=np.float32, copy=False)
        return (arr - self.mean) / self.scale

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with self._torch.no_grad():
            xt = self._torch.from_numpy(self._transform(x)).to(self.device)
            out = self.model(xt).squeeze(1)
            p = out.detach().cpu().numpy()
        return p


def _feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {
        "timestamp",
        "date",
        "future_ret",
        "label",
        "target_leverage",
        "open_time",
        "close_time",
        "equity_curve_proxy",
        "rolling_peak",
    }
    cols = [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def _clean_xy(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    x = x.ffill().bfill()
    x = x.fillna(0)
    return x


def _build_cls_mlp(torch_mod: Any, in_dim: int, out_dim: int) -> Any:
    return torch_mod.nn.Sequential(
        torch_mod.nn.Linear(in_dim, 256),
        torch_mod.nn.ReLU(),
        torch_mod.nn.Dropout(0.05),
        torch_mod.nn.Linear(256, 128),
        torch_mod.nn.ReLU(),
        torch_mod.nn.Linear(128, out_dim),
    )


def _build_reg_mlp(torch_mod: Any, in_dim: int) -> Any:
    return torch_mod.nn.Sequential(
        torch_mod.nn.Linear(in_dim, 256),
        torch_mod.nn.ReLU(),
        torch_mod.nn.Dropout(0.05),
        torch_mod.nn.Linear(256, 128),
        torch_mod.nn.ReLU(),
        torch_mod.nn.Linear(128, 1),
    )


def _resolve_torch_device(requested: str) -> tuple[Any, str, Any]:
    try:
        import torch  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"PyTorch not available: {e}") from e

    req = str(requested or "auto").lower()

    if req in {"directml", "npu"}:
        try:
            import torch_directml  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"DirectML backend not available: {e}") from e
        return torch_directml.device(), "directml", torch

    if req in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda", torch
        raise RuntimeError("CUDA backend not available on this machine.")

    if req == "cpu":
        return torch.device("cpu"), "cpu", torch

    # auto / cloud: prefer cloud-style accelerators first.
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", torch
    try:
        import torch_directml  # type: ignore
        return torch_directml.device(), "directml", torch
    except Exception:
        pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps", torch
    return torch.device("cpu"), "cpu", torch


def _fit_torch_accelerated(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    settings: Settings,
    requested_device: str,
    progress_cb: Callable[[int, str], None] | None = None,
) -> Tuple[TrainedModels, dict]:
    device, resolved_device_name, torch = _resolve_torch_device(requested_device)

    x_train = _clean_xy(train_df, feature_cols).to_numpy(dtype=np.float32, copy=False)
    x_test = _clean_xy(test_df, feature_cols).to_numpy(dtype=np.float32, copy=False)
    y_train = train_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    x_test_s = scaler.transform(x_test).astype(np.float32, copy=False)
    mean = scaler.mean_.astype(np.float32)
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)

    class_values = np.array(sorted(set(y_train.tolist()) | set(y_test.tolist())), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(class_values)}
    y_train_idx = np.array([class_to_idx[v] for v in y_train], dtype=np.int64)

    in_dim = x_train_s.shape[1]
    n_classes = len(class_values)
    batch_size = max(64, int(settings.torch_batch_size))
    epochs = max(3, int(settings.torch_epochs))

    cls_model = _build_cls_mlp(torch, in_dim, n_classes).to(device)
    cls_opt = torch.optim.AdamW(cls_model.parameters(), lr=1e-3, weight_decay=1e-4)
    cls_loss = torch.nn.CrossEntropyLoss()

    n_train = len(x_train_s)
    for ep in range(epochs):
        if progress_cb:
            progress_cb(66 + int(ep / epochs * 7), f"訓練分類模型（Epoch {ep+1}/{epochs}）")
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = torch.from_numpy(x_train_s[idx]).to(device)
            yb = torch.from_numpy(y_train_idx[idx]).to(device)
            cls_opt.zero_grad()
            logits = cls_model(xb)
            loss = cls_loss(logits, yb)
            loss.backward()
            cls_opt.step()

    with torch.no_grad():
        cls_model.eval()
        xt = torch.from_numpy(x_test_s).to(device)
        proba = torch.softmax(cls_model(xt), dim=1).detach().cpu().numpy()
    y_pred = class_values[np.argmax(proba, axis=1)]
    cls_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    lev_train = train_df["target_leverage"].clip(1, settings.max_leverage).to_numpy(dtype=np.float32)
    lev_test = test_df["target_leverage"].clip(1, settings.max_leverage).to_numpy(dtype=np.float32)

    reg_model = _build_reg_mlp(torch, in_dim).to(device)
    reg_opt = torch.optim.AdamW(reg_model.parameters(), lr=1e-3, weight_decay=1e-4)
    reg_loss = torch.nn.MSELoss()

    for ep in range(max(3, epochs // 2)):
        if progress_cb:
            progress_cb(73 + int(ep / max(3, epochs // 2) * 7), f"訓練槓桿模型（Epoch {ep+1}/{max(3, epochs // 2)}）")
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = torch.from_numpy(x_train_s[idx]).to(device)
            yb = torch.from_numpy(lev_train[idx]).to(device)
            reg_opt.zero_grad()
            pred = reg_model(xb).squeeze(1)
            loss = reg_loss(pred, yb)
            loss.backward()
            reg_opt.step()

    with torch.no_grad():
        reg_model.eval()
        xt = torch.from_numpy(x_test_s).to(device)
        lev_pred = reg_model(xt).squeeze(1).detach().cpu().numpy()
    lev_mae = mean_absolute_error(lev_test, lev_pred)

    models = TrainedModels(
        clf=_TorchSignalWrapper(cls_model, device, mean, scale, class_values, torch),
        lev_reg=_TorchLevWrapper(reg_model, device, mean, scale, torch),
        feature_cols=feature_cols,
        backend="torch_accel",
        backend_meta={"device": str(device), "in_dim": int(in_dim), "class_values": class_values.tolist(), "mean": mean.tolist(), "scale": scale.tolist()},
    )
    metrics = {
        "classification_report": cls_report,
        "leverage_mae": float(lev_mae),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "training_backend": "torch_accel",
        "training_device": str(device),
        "training_device_kind": resolved_device_name,
    }
    return models, metrics


def _fit_sklearn_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    settings: Settings,
    progress_cb: Callable[[int, str], None] | None = None,
    soft_labels_df: pd.DataFrame | None = None,
    distill_alpha: float = 0.4,
) -> Tuple[TrainedModels, dict]:
    """
    distill_alpha: Teacher 頠?蝐斗毽?交?靘?.0=蝝′璅惜, 1.0=蝝?璅惜
    撖阡??剖??孵?: y_blend = (1-alpha)*hard + alpha*soft
    """
    x_train = _clean_xy(train_df, feature_cols)
    y_train_hard = train_df["label"].to_numpy(dtype=int)
    x_test = _clean_xy(test_df, feature_cols)
    y_test = test_df["label"]

    # ?? ?賊冗瘛瑕?璅惜 ???????????????????????????????????????????
    distill_applied = False
    sample_weight: np.ndarray | None = None

    if soft_labels_df is not None and not soft_labels_df.empty and distill_alpha > 0:
        # 撠? timestamp 嚗oft_labels ?航瘨菔??券?嚗?閮毀???停憟踝?
        _sl = soft_labels_df.copy()
        _has_ts = "timestamp" in _sl.columns and "timestamp" in train_df.columns

        soft_p_long  = np.full(len(train_df), 1/3, dtype=np.float64)
        soft_p_flat  = np.full(len(train_df), 1/3, dtype=np.float64)
        soft_p_short = np.full(len(train_df), 1/3, dtype=np.float64)

        if _has_ts:
            _sl_idx = pd.to_datetime(_sl["timestamp"], utc=True, errors="coerce")
            _sl = _sl.set_index(_sl_idx)
            _tr_ts = pd.to_datetime(train_df["timestamp"], utc=True, errors="coerce")
            for i, ts in enumerate(_tr_ts):
                if ts in _sl.index:
                    row = _sl.loc[ts]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    if "soft_p_long" in _sl.columns:
                        soft_p_long[i]  = float(row.get("soft_p_long",  1/3))
                        soft_p_flat[i]  = float(row.get("soft_p_flat",  1/3))
                        soft_p_short[i] = float(row.get("soft_p_short", 1/3))
        else:
            # 雿蔭撠??航???蝑摰?詨?嚗?            n_match = min(len(train_df), len(_sl))
            if "soft_p_long" in _sl.columns:
                soft_p_long[:n_match]  = _sl["soft_p_long"].values[:n_match]
                soft_p_flat[:n_match]  = _sl["soft_p_flat"].values[:n_match]
                soft_p_short[:n_match] = _sl["soft_p_short"].values[:n_match]

        # Teacher soft probabilities -> class index.
        soft_class_arr = np.array([soft_p_long, soft_p_flat, soft_p_short], dtype=np.float64)  # (3, N)
        soft_pred_idx = np.argmax(soft_class_arr, axis=0)  # 0=long, 1=flat, 2=short
        class_map = {0: 1, 1: 0, 2: -1}  # 頧? label 撠望頠?蝐斤?銝餃撐
        y_soft = np.array([class_map[i] for i in soft_pred_idx], dtype=int)
        teacher_conf = np.max(soft_class_arr, axis=0)  # 靽∪?摨?
        # sample_weight: 蝖祆?蝐文?頠?蝐支?????甈? 1.0嚗?銝??????alpha*teacher_conf 瘙箏?
        agree = (y_train_hard == y_soft)
        sample_weight = np.where(
            agree,
            1.0,
            distill_alpha * teacher_conf + (1.0 - distill_alpha) * (1.0 - teacher_conf)
        )
        sample_weight = (sample_weight / sample_weight.mean()).clip(0.2, 3.0)  # 璅???
        # y_train ?寧頠?蝐文
        y_train_mixed = np.where(agree, y_train_hard, y_soft)
        y_train = pd.Series(y_train_mixed)
        distill_applied = True
        if progress_cb:
            agree_pct = agree.mean() * 100
            progress_cb(63, f"蒸餾對齊完成：Teacher/Student 一致率 {agree_pct:.1f}%")
    else:
        y_train = pd.Series(y_train_hard)

    if progress_cb:
        progress_cb(68, "訓練分類模型（RandomForest" + (" + Teacher 蒸餾" if distill_applied else "") + "）")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(x_train, y_train, sample_weight=sample_weight)

    y_pred = clf.predict(x_test)
    cls_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    lev_train = train_df["target_leverage"].clip(1, settings.max_leverage)
    lev_test  = test_df["target_leverage"].clip(1, settings.max_leverage)

    # ?? ?賊冗瑽▼瘛瑕? ?????????????????????????????????????????
    if distill_applied and soft_labels_df is not None and "teacher_leverage" in soft_labels_df.columns:
        _has_ts2 = "timestamp" in soft_labels_df.columns and "timestamp" in train_df.columns
        soft_lev = lev_train.to_numpy(dtype=np.float64).copy()
        if _has_ts2:
            _sl2 = soft_labels_df.set_index(
                pd.to_datetime(soft_labels_df["timestamp"], utc=True, errors="coerce")
            )
            _tr_ts2 = pd.to_datetime(train_df["timestamp"], utc=True, errors="coerce")
            for i, ts in enumerate(_tr_ts2):
                if ts in _sl2.index:
                    row2 = _sl2.loc[ts]
                    if isinstance(row2, pd.DataFrame):
                        row2 = row2.iloc[0]
                    soft_lev[i] = float(row2.get("teacher_leverage", lev_train.iloc[i]))
        lev_train_arr = ((1 - distill_alpha) * lev_train.to_numpy() + distill_alpha * soft_lev)
    else:
        lev_train_arr = lev_train.to_numpy(dtype=np.float64)
    lev_train_arr = np.clip(lev_train_arr, 1.0, float(settings.max_leverage))

    asym_w = _asymmetric_lev_weights(lev_train_arr, penalty_factor=3.0)
    if sample_weight is not None:
        lev_sample_w = (sample_weight * asym_w)
        lev_sample_w = (lev_sample_w / lev_sample_w.mean()).clip(0.2, 5.0)
    else:
        lev_sample_w = asym_w

    # ?? 撅?2嚗??典翰??RF ??warm-up ?葫嚗???Pinball 甈?蝎曄毀 ????
    if progress_cb:
        progress_cb(74, "訓練槓桿模型（RF warm-up）")
    rf_warm = RandomForestRegressor(
        n_estimators=50,   # 敹恍?warm-up
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_warm.fit(x_train, lev_train_arr, sample_weight=lev_sample_w)
    lev_pred_warm = rf_warm.predict(x_train)

    # Pinball-style weighting: penalize over-estimated leverage more.
    pinball_w = _pinball_sample_weights(lev_train_arr, lev_pred_warm, tau=0.35)
    final_lev_w = (lev_sample_w * pinball_w)
    final_lev_w = (final_lev_w / final_lev_w.mean()).clip(0.15, 6.0)

    # ?? 撅?3嚗 GradientBoosting Quantile ?飛嚗撱粹?撠迂 loss嚗???
    if progress_cb:
        progress_cb(78, "訓練槓桿模型（GB Quantile loss, tau=0.35）")
    gb_lev = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        loss="quantile",      # ?批遣 Pinball loss
        alpha=0.35,           # ?葫蝚?35 ?曉?雿???憭拍?靽???        random_state=42,
    )
    gb_lev.fit(x_train, lev_train_arr, sample_weight=final_lev_w)

    # RF leverage warm model
    lev_reg_rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    )
    lev_reg_rf.fit(x_train, lev_train_arr, sample_weight=final_lev_w)
    lev_reg = EnsembleLevReg(lev_reg_rf, gb_lev, rf_w=0.40)

    # Evaluate leverage regression on the test split.
    lev_pred = lev_reg.predict(x_test)
    lev_mae = mean_absolute_error(lev_test.to_numpy(), lev_pred)
    overestimate_rate = float((lev_pred > lev_test.to_numpy()).mean())

    models = TrainedModels(
        clf=clf,
        lev_reg=lev_reg,
        feature_cols=feature_cols,
        backend="sklearn_rf",
        backend_meta={
            "device": "cpu",
            "distilled": distill_applied,
            "lev_loss": "quantile_tau035+asymmetric_weight",
            "overestimate_rate": overestimate_rate,
        },
    )
    metrics = {
        "classification_report": cls_report,
        "leverage_mae": float(lev_mae),
        "leverage_overestimate_rate": overestimate_rate,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "training_backend": "sklearn_rf",
        "training_device": "cpu",
        "distillation_applied": distill_applied,
        "distill_alpha": distill_alpha if distill_applied else 0.0,
        "lev_loss": "quantile_tau035+asymmetric_weight",
    }
    return models, metrics


def train_models(
    df: pd.DataFrame,
    settings: Settings,
    progress_cb: Callable[[int, str], None] | None = None,
    soft_labels_df: pd.DataFrame | None = None,
    distill_alpha: float = 0.4,
) -> Tuple[TrainedModels, dict]:
    # soft_labels_df: Teacher soft-label DataFrame with timestamp and soft probabilities.
    # distill_alpha: 0.4 means 60% hard labels + 40% teacher guidance.
    max_rows = int(getattr(settings, "max_train_rows", 0) or 0)
    if max_rows > 0 and len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)

    if len(df) < settings.min_train_rows:
        raise RuntimeError(f"Not enough rows for training. Need >= {settings.min_train_rows}, got {len(df)}")

    feature_cols = _feature_columns(df)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    requested = str(settings.train_device or "auto").lower()

    # ?? CPU 璅∪?嚗?渲擗???????????????????????????????????
    if requested == "cpu":
        return _fit_sklearn_rf(
            train_df, test_df, feature_cols, settings, progress_cb,
            soft_labels_df=soft_labels_df, distill_alpha=distill_alpha,
        )

    # ?? ?芋撘??岫 torch嚗仃?? fallback ??sklearn嚗?賊冗嚗????????
    if requested in {"auto", "cloud", "npu", "directml", "cuda", "gpu", "mps"}:
        try:
            return _fit_torch_accelerated(train_df, test_df, feature_cols, settings, requested, progress_cb)
        except Exception as e:  # noqa: BLE001
            if requested in {"npu", "directml", "cuda", "gpu", "mps"} and settings.npu_strict:
                raise RuntimeError(f"Accelerated mode enabled, but accelerator training failed: {e}") from e
            models, metrics = _fit_sklearn_rf(
                train_df, test_df, feature_cols, settings, progress_cb,
                soft_labels_df=soft_labels_df, distill_alpha=distill_alpha,
            )
            metrics["training_note"] = f"Accelerator unavailable, fallback to CPU (+distill): {e}"
            return models, metrics

    return _fit_sklearn_rf(
        train_df, test_df, feature_cols, settings, progress_cb,
        soft_labels_df=soft_labels_df, distill_alpha=distill_alpha,
    )


def save_models(models: TrainedModels, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    if str(models.backend).startswith("torch"):
        try:
            import torch  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"torch is required to save torch models: {e}") from e

        bundle = {
            "backend": models.backend,
            "feature_cols": models.feature_cols,
            "backend_meta": models.backend_meta or {},
            "clf_state_dict": models.clf.model.state_dict(),
            "lev_state_dict": models.lev_reg.model.state_dict(),
        }
        torch.save(bundle, model_dir / "torch_models.pt")
        return

    joblib.dump(models.clf, model_dir / "signal_clf.joblib")
    joblib.dump(models.lev_reg, model_dir / "leverage_reg.joblib")
    joblib.dump(models.feature_cols, model_dir / "feature_cols.joblib")


def load_models(model_dir: Path) -> TrainedModels:
    torch_bundle_path = model_dir / "torch_models.pt"
    clf_path = model_dir / "signal_clf.joblib"
    lev_path = model_dir / "leverage_reg.joblib"
    feat_path = model_dir / "feature_cols.joblib"
    _has_sklearn = clf_path.exists() and lev_path.exists() and feat_path.exists()

    if torch_bundle_path.exists():
        try:
            import torch  # type: ignore
        except Exception as torch_err:  # noqa: BLE001
            if _has_sklearn:
                import warnings
                warnings.warn(
                    f"[load_models] torch unavailable ({torch_err}); using sklearn fallback.",
                    stacklevel=2,
                )
                clf = joblib.load(clf_path)
                lev_reg = joblib.load(lev_path)
                feature_cols = joblib.load(feat_path)
                return TrainedModels(
                    clf=clf, lev_reg=lev_reg, feature_cols=feature_cols,
                    backend="sklearn_rf",
                    backend_meta={"device": "cpu", "note": "torch_unavailable_sklearn_fallback"},
                )
            raise RuntimeError(
                f"torch is required but unavailable: {torch_err}\n"
                "Use the local .venv311 to run the dashboard or install a torch-compatible environment.\n"
                "If you only need inference, keep the sklearn .joblib models available.",
            ) from torch_err

        # ?? torch ?舐嚗迤撣貉???torch bundle ???????????????????????????
        bundle = torch.load(torch_bundle_path, map_location="cpu")
        backend_name = str(bundle.get("backend", "torch_accel"))
        meta = bundle.get("backend_meta") or {}
        feature_cols = list(bundle.get("feature_cols") or [])
        in_dim = int(meta.get("in_dim", 0))
        class_values = np.array(meta.get("class_values", [-1, 0, 1]), dtype=int)
        mean = np.array(meta.get("mean", []), dtype=np.float32)
        scale = np.array(meta.get("scale", []), dtype=np.float32)

        device_str = str(meta.get("device", "")).lower()
        device = torch.device("cpu")
        if "cuda" in device_str and torch.cuda.is_available():
            device = torch.device("cuda")
        elif "mps" in device_str and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif "privateuseone" in device_str or "directml" in device_str:
            try:
                import torch_directml  # type: ignore
                device = torch_directml.device()
            except Exception:
                device = torch.device("cpu")

        if in_dim <= 0:
            raise RuntimeError("Invalid torch model bundle: missing input dimension.")

        cls_model = _build_cls_mlp(torch, in_dim, len(class_values))
        reg_model = _build_reg_mlp(torch, in_dim)
        cls_model.load_state_dict(bundle["clf_state_dict"])
        reg_model.load_state_dict(bundle["lev_state_dict"])
        cls_model = cls_model.to(device)
        reg_model = reg_model.to(device)

        return TrainedModels(
            clf=_TorchSignalWrapper(cls_model, device, mean, scale, class_values, torch),
            lev_reg=_TorchLevWrapper(reg_model, device, mean, scale, torch),
            feature_cols=feature_cols,
            backend=backend_name,
            backend_meta=meta,
        )

    if not _has_sklearn:
        raise FileNotFoundError(
            "No torch bundle was found, and the fallback sklearn .joblib files are missing.",
        )

    clf = joblib.load(clf_path)
    lev_reg = joblib.load(lev_path)
    feature_cols = joblib.load(feat_path)
    return TrainedModels(
        clf=clf, lev_reg=lev_reg, feature_cols=feature_cols,
        backend="sklearn_rf", backend_meta={"device": "cpu"},
    )


def infer_signals(df: pd.DataFrame, models: TrainedModels, settings: Settings) -> pd.DataFrame:
    x = _clean_xy(df, models.feature_cols)
    proba = models.clf.predict_proba(x)
    classes = list(models.clf.classes_)

    idx_map = {c: i for i, c in enumerate(classes)}
    p_long = proba[:, idx_map.get(1, 0)] if 1 in idx_map else np.zeros(len(df))
    p_short = proba[:, idx_map.get(-1, 0)] if -1 in idx_map else np.zeros(len(df))
    p_flat = proba[:, idx_map.get(0, 0)] if 0 in idx_map else np.zeros(len(df))

    signal = np.where((p_long > 0.45) & (p_long > p_short), 1, np.where((p_short > 0.45) & (p_short > p_long), -1, 0))

    raw_lev = models.lev_reg.predict(x)
    confidence = np.maximum(p_long, p_short) - p_flat
    conf_scale = np.clip(confidence * 2.0, 0.2, 1.0)

    max_safe_lev = compute_max_safe_leverage(df, settings.max_leverage)
    leverage = np.clip(raw_lev * conf_scale, 1, settings.max_leverage)
    leverage = np.minimum(leverage, max_safe_lev)

    out = df.copy()
    out["p_long"] = p_long
    out["p_short"] = p_short
    out["p_flat"] = p_flat
    out["signal"] = signal
    out["suggested_leverage"] = leverage.round(2)
    out["max_safe_leverage"] = max_safe_lev.round(2)

    # ?? AI 靽∪?? & 撣憸冽嚗???K 蝺???? ?????????????????????
    confidence_index = np.maximum(p_long + p_short - p_flat, 0.0)
    out["confidence_index"] = confidence_index.round(4)

    def _classify_row_style(idx: int) -> str:
        fg = float(out.at[idx, "fear_greed_value"]) if "fear_greed_value" in out.columns else 50.0
        vol24 = float(out.at[idx, "realized_vol_24"]) if "realized_vol_24" in out.columns else 0.03
        atr_p = float(out.at[idx, "atr_pct"]) if "atr_pct" in out.columns else 0.015
        conf = float(confidence_index[idx])
        macd_h = float(out.at[idx, "macd_hist"]) if "macd_hist" in out.columns else 0.0
        dd = float(out.at[idx, "drawdown"]) if "drawdown" in out.columns else 0.0

        s = 0.0
        if fg >= 75:   s += 1.2
        elif fg >= 55: s += 0.6
        elif fg <= 25: s -= 1.5
        elif fg <= 40: s -= 0.7
        if vol24 < 0.015:  s += 0.8
        elif vol24 < 0.025: s += 0.3
        elif vol24 > 0.06:  s -= 1.2
        elif vol24 > 0.04:  s -= 0.6
        if atr_p < 0.008:  s += 0.5
        elif atr_p > 0.025: s -= 0.8
        if conf >= 0.35:   s += 0.8
        elif conf >= 0.20: s += 0.3
        elif conf < 0.05:  s -= 0.5
        if macd_h != 0:    s += 0.4 * (1 if macd_h > 0 else -1)
        if dd < -0.15:     s -= 1.0
        elif dd < -0.08:   s -= 0.5

        s = max(-3.0, min(3.0, s))
        if s >= 0.8:
            return "aggressive"
        elif s <= -0.6:
            return "conservative"
        else:
            return "neutral"

    out["ai_style"] = [_classify_row_style(i) for i in range(len(out))]
    return out



def compute_max_safe_leverage(df: pd.DataFrame, hard_cap: int) -> np.ndarray:
    atr_pct = df["atr_pct"].replace(0, np.nan).ffill().fillna(0.01)
    vol = df["realized_vol_24"].replace(0, np.nan).ffill().fillna(0.03)
    drawdown = df["drawdown"].abs().fillna(0)

    # Lower volatility and lower drawdown allow higher leverage.
    base = 0.03 / (atr_pct + vol)
    dd_penalty = np.clip(1 - drawdown * 2.5, 0.1, 1.0)
    lev = base * dd_penalty * 12
    lev = np.clip(lev, 1, hard_cap)
    return lev.to_numpy()
