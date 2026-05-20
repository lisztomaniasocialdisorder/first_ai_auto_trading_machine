"""
src/distillation.py
===================
知識蒸餾模組：從現有 CSV 訓練 Teacher 大模型，產生軟標籤供 Student 學習。

Teacher 架構（純 sklearn，不需 torch）：
  - 分類器：RandomForest(500棵) + GradientBoostingClassifier 加權集成
  - 回歸器：RandomForest(500棵) + GradientBoostingRegressor 加權集成
  - Temperature Scaling：軟化機率分佈（T > 1 更模糊，T < 1 更銳利）

輸出：
  1. models/teacher/{symbol}_{interval}/ — teacher 模型檔
  2. outputs/teacher_soft_labels_{symbol}_{interval}.csv — 每根 K 線的軟標籤
  3. outputs/teacher_report_{symbol}_{interval}.json — 訓練報告
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ── 常數 ────────────────────────────────────────────────────────────────────
BLOCKED_COLS = {
    "timestamp", "date", "future_ret", "label", "target_leverage",
    "open_time", "close_time", "equity_curve_proxy", "rolling_peak",
    # 排除已知預測欄位，避免 data leakage
    "p_long", "p_short", "p_flat", "signal",
    "suggested_leverage", "max_safe_leverage", "confidence_index", "ai_style",
}

TEACHER_VERSION = "1.0"


# ── 工具函數 ─────────────────────────────────────────────────────────────────
def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in BLOCKED_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def _clean_xy(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    x = x.ffill().bfill().fillna(0)
    return x.to_numpy(dtype=np.float64)


def _temperature_scale(proba: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    """
    Soft-max 溫度縮放：T > 1 → 機率更平滑（適合 student 學習）
    logits = log(p) / T  → renormalize
    """
    temperature = max(temperature, 0.01)
    log_p = np.log(np.clip(proba, 1e-9, 1.0)) / temperature
    log_p -= log_p.max(axis=1, keepdims=True)   # 數值穩定
    exp_p = np.exp(log_p)
    return exp_p / exp_p.sum(axis=1, keepdims=True)


def _ensemble_proba(
    rf_proba: np.ndarray,
    gb_proba: np.ndarray,
    rf_weight: float = 0.55,
) -> np.ndarray:
    """加權平均兩個分類器的機率輸出。"""
    return rf_weight * rf_proba + (1.0 - rf_weight) * gb_proba


def _ensemble_reg(
    rf_pred: np.ndarray,
    gb_pred: np.ndarray,
    rf_weight: float = 0.55,
) -> np.ndarray:
    return rf_weight * rf_pred + (1.0 - rf_weight) * gb_pred


# ── Teacher 訓練主函數 ────────────────────────────────────────────────────────
def train_teacher(
    csv_path: Path,
    model_dir: Path,
    output_dir: Path,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    max_rows: int = 0,
    temperature: float = 2.0,
    rf_weight: float = 0.55,
    n_rf_estimators: int = 500,
    gb_n_estimators: int = 200,
    progress_cb: Callable[[int, str], None] | None = None,
) -> dict:
    """
    從 signals_with_features CSV 訓練 Teacher 集成模型，產生軟標籤。

    Parameters
    ----------
    csv_path       : signals_with_features_{symbol}_{interval}.csv 路徑
    model_dir      : teacher 模型儲存目錄
    output_dir     : 軟標籤 CSV & report 儲存目錄
    symbol / interval : 交易對與週期
    max_rows       : 0 = 全量；> 0 = 只用最近 N 筆
    temperature    : 軟標籤溫度（建議 1.5~3.0）
    rf_weight      : RF 在集成中的比重（GradientBoosting = 1 - rf_weight）
    n_rf_estimators: RandomForest 樹數
    gb_n_estimators: GradientBoosting 迭代數

    Returns
    -------
    dict: 訓練報告
    """

    def _cb(p: int, msg: str) -> None:
        if progress_cb:
            progress_cb(p, msg)

    # ── 1. 載入資料 ──────────────────────────────────────────────────────────
    _cb(5, "載入 CSV 資料...")
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True) if "timestamp" in df.columns else df

    if max_rows > 0 and len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)

    # 過濾掉沒有真實標籤的列（future_ret 為 NaN 表示未來資料不足）
    if "label" in df.columns:
        df = df[df["label"].notna()].reset_index(drop=True)
    if "target_leverage" in df.columns:
        df = df[df["target_leverage"].notna()].reset_index(drop=True)

    n_total = len(df)
    _cb(10, f"資料筆數：{n_total:,}，開始特徵選取...")

    if n_total < 200:
        raise RuntimeError(f"資料太少（{n_total} 筆），至少需要 200 筆才能訓練 Teacher。")

    # ── 2. 特徵與標籤 ─────────────────────────────────────────────────────────
    feature_cols = _feature_columns(df)
    _cb(12, f"特徵數：{len(feature_cols)} 個")

    X = _clean_xy(df, feature_cols)
    y_cls = df["label"].astype(int).to_numpy()
    y_lev = df["target_leverage"].clip(1, 100).to_numpy(dtype=np.float64)

    class_values = np.array(sorted(set(y_cls.tolist())), dtype=int)

    # 80/20 切分用於評估（Teacher 也需要做 OOF 評估才知道品質）
    split = int(n_total * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_cls_train, y_cls_test = y_cls[:split], y_cls[split:]
    y_lev_train, y_lev_test = y_lev[:split], y_lev[split:]

    # 標準化（GradientBoosting 不需要，但保留 scaler 供 student 參考）
    scaler = StandardScaler()
    scaler.fit(X_train)   # 只 fit 訓練集

    _cb(15, "開始訓練 Teacher RandomForest（分類）...")

    # ── 3. 分類器訓練 ─────────────────────────────────────────────────────────
    rf_clf = RandomForestClassifier(
        n_estimators=n_rf_estimators,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf_clf.fit(X_train, y_cls_train)
    _cb(35, "RandomForest(分類) 訓練完成，訓練 GradientBoosting...")

    gb_clf = GradientBoostingClassifier(
        n_estimators=gb_n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    gb_clf.fit(X_train, y_cls_train)
    _cb(55, "GradientBoosting(分類) 訓練完成，開始回歸器...")

    # ── 4. 回歸器訓練（槓桿） ─────────────────────────────────────────────────
    rf_reg = RandomForestRegressor(
        n_estimators=n_rf_estimators,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf_reg.fit(X_train, y_lev_train)
    _cb(70, "RandomForest(槓桿) 訓練完成，訓練 GradientBoosting(槓桿)...")

    gb_reg = GradientBoostingRegressor(
        n_estimators=gb_n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        loss="quantile",   # 非對稱 Pinball loss，天生保守
        alpha=0.35,        # 預測第35百分位 → 寧低估不高估
        random_state=42,
    )
    gb_reg.fit(X_train, y_lev_train)
    _cb(80, "回歸器訓練完成（quantile tau=0.35），產生軟標籤...")

    # ── 5. 生成全量軟標籤（在全部資料上推理，包含訓練集） ──────────────────────
    # 全量特徵
    X_all = _clean_xy(df, feature_cols)

    rf_proba_all = rf_clf.predict_proba(X_all)    # shape: (N, n_classes)
    gb_proba_all = gb_clf.predict_proba(X_all)

    # 對齊 class 順序（RF 和 GB 的 classes_ 可能順序不同）
    def _align_proba(clf, proba: np.ndarray) -> np.ndarray:
        """確保機率欄對齊 class_values 順序。"""
        cls_order = list(clf.classes_)
        out = np.zeros((len(proba), len(class_values)), dtype=np.float64)
        for i, c in enumerate(class_values):
            if c in cls_order:
                out[:, i] = proba[:, cls_order.index(c)]
        return out

    rf_proba_all = _align_proba(rf_clf, rf_proba_all)
    gb_proba_all = _align_proba(gb_clf, gb_proba_all)

    raw_ensemble_proba = _ensemble_proba(rf_proba_all, gb_proba_all, rf_weight)
    soft_proba = _temperature_scale(raw_ensemble_proba, temperature)

    rf_lev_all = rf_reg.predict(X_all)
    gb_lev_all = gb_reg.predict(X_all)
    soft_leverage = _ensemble_reg(rf_lev_all, gb_lev_all, rf_weight)
    soft_leverage = np.clip(soft_leverage, 1.0, 100.0)

    # ── 6. 測試集評估 ──────────────────────────────────────────────────────────
    X_test_all = X_test   # 已是 numpy
    rf_p_test = _align_proba(rf_clf, rf_clf.predict_proba(X_test_all))
    gb_p_test = _align_proba(gb_clf, gb_clf.predict_proba(X_test_all))
    ens_p_test = _ensemble_proba(rf_p_test, gb_p_test, rf_weight)
    y_pred_test = class_values[np.argmax(ens_p_test, axis=1)]
    cls_rpt = classification_report(y_cls_test, y_pred_test, output_dict=True, zero_division=0)

    rf_lev_test = rf_reg.predict(X_test_all)
    gb_lev_test = gb_reg.predict(X_test_all)
    ens_lev_test = _ensemble_reg(rf_lev_test, gb_lev_test, rf_weight)
    lev_mae = mean_absolute_error(y_lev_test, ens_lev_test)

    _cb(88, "評估完成，儲存模型與軟標籤...")

    # ── 7. 儲存 Teacher 模型 ──────────────────────────────────────────────────
    tag = f"{symbol}_{interval}"
    t_dir = model_dir / "teacher" / tag
    t_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf_clf, t_dir / "teacher_rf_clf.joblib")
    joblib.dump(gb_clf, t_dir / "teacher_gb_clf.joblib")
    joblib.dump(rf_reg, t_dir / "teacher_rf_reg.joblib")
    joblib.dump(gb_reg, t_dir / "teacher_gb_reg.joblib")
    joblib.dump(scaler, t_dir / "teacher_scaler.joblib")
    joblib.dump(feature_cols, t_dir / "teacher_feature_cols.joblib")

    # 儲存 metadata
    meta = {
        "version": TEACHER_VERSION,
        "symbol": symbol,
        "interval": interval,
        "n_rows": int(n_total),
        "n_features": len(feature_cols),
        "class_values": class_values.tolist(),
        "temperature": float(temperature),
        "rf_weight": float(rf_weight),
        "n_rf_estimators": int(n_rf_estimators),
        "gb_n_estimators": int(gb_n_estimators),
        "trained_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    (t_dir / "teacher_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── 8. 儲存軟標籤 CSV ────────────────────────────────────────────────────
    out_df = df[["timestamp"]].copy() if "timestamp" in df.columns else pd.DataFrame(index=df.index)

    # 軟標籤機率（溫度縮放後）
    for i, c in enumerate(class_values):
        col_name = {-1: "soft_p_short", 0: "soft_p_flat", 1: "soft_p_long"}.get(int(c), f"soft_p_{c}")
        out_df[col_name] = soft_proba[:, i].round(6)

    # 原始集成機率（未溫度縮放）
    for i, c in enumerate(class_values):
        col_name = {-1: "raw_p_short", 0: "raw_p_flat", 1: "raw_p_long"}.get(int(c), f"raw_p_{c}")
        out_df[col_name] = raw_ensemble_proba[:, i].round(6)

    out_df["teacher_signal"] = class_values[np.argmax(raw_ensemble_proba, axis=1)]
    out_df["teacher_leverage"] = soft_leverage.round(4)
    out_df["teacher_confidence"] = np.max(raw_ensemble_proba, axis=1).round(6)

    # 保留真實標籤供對比
    if "label" in df.columns:
        out_df["true_label"] = df["label"].values
    if "target_leverage" in df.columns:
        out_df["true_leverage"] = df["target_leverage"].round(4).values

    soft_label_path = output_dir / f"teacher_soft_labels_{tag}.csv"
    out_df.to_csv(soft_label_path, index=False, encoding="utf-8")

    # ── 9. 儲存報告 ───────────────────────────────────────────────────────────
    report = {
        "meta": meta,
        "classification_report": cls_rpt,
        "leverage_mae": float(lev_mae),
        "soft_label_path": str(soft_label_path),
        "model_dir": str(t_dir),
        "soft_label_stats": {
            "mean_soft_p_long": float(out_df.get("soft_p_long", pd.Series([0])).mean()),
            "mean_soft_p_short": float(out_df.get("soft_p_short", pd.Series([0])).mean()),
            "mean_soft_p_flat": float(out_df.get("soft_p_flat", pd.Series([0])).mean()),
            "mean_teacher_confidence": float(out_df["teacher_confidence"].mean()),
            "mean_teacher_leverage": float(out_df["teacher_leverage"].mean()),
        },
    }

    rpt_path = output_dir / f"teacher_report_{tag}.json"
    rpt_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _cb(100, f"Teacher 訓練完成！軟標籤已儲存到 {soft_label_path.name}")
    return report


# ── 載入 Teacher 做推理 ────────────────────────────────────────────────────
def load_teacher_and_infer(
    df: pd.DataFrame,
    model_dir: Path,
    symbol: str,
    interval: str,
    temperature: float | None = None,
    rf_weight: float | None = None,
) -> pd.DataFrame:
    """
    載入已儲存的 Teacher，對 df 做推理，回傳加入軟標籤欄位的 DataFrame。
    """
    tag = f"{symbol}_{interval}"
    t_dir = model_dir / "teacher" / tag

    if not (t_dir / "teacher_meta.json").exists():
        raise FileNotFoundError(
            f"找不到 Teacher 模型：{t_dir}\n"
            "請先執行「訓練 Teacher 模型」。"
        )

    meta = json.loads((t_dir / "teacher_meta.json").read_text(encoding="utf-8"))
    temperature = temperature if temperature is not None else float(meta.get("temperature", 2.0))
    rf_weight = rf_weight if rf_weight is not None else float(meta.get("rf_weight", 0.55))
    class_values = np.array(meta["class_values"], dtype=int)

    rf_clf = joblib.load(t_dir / "teacher_rf_clf.joblib")
    gb_clf = joblib.load(t_dir / "teacher_gb_clf.joblib")
    rf_reg = joblib.load(t_dir / "teacher_rf_reg.joblib")
    gb_reg = joblib.load(t_dir / "teacher_gb_reg.joblib")
    feature_cols: list[str] = joblib.load(t_dir / "teacher_feature_cols.joblib")

    # 特徵對齊
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        warnings.warn(f"Teacher 推理：缺少 {len(missing)} 個特徵欄位，補 0。", stacklevel=2)

    x_df = df.reindex(columns=feature_cols, fill_value=0)
    X = x_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0).to_numpy(dtype=np.float64)

    def _align(clf, proba):
        cls_order = list(clf.classes_)
        out = np.zeros((len(proba), len(class_values)), dtype=np.float64)
        for i, c in enumerate(class_values):
            if c in cls_order:
                out[:, i] = proba[:, cls_order.index(c)]
        return out

    rf_proba = _align(rf_clf, rf_clf.predict_proba(X))
    gb_proba = _align(gb_clf, gb_clf.predict_proba(X))
    raw_proba = _ensemble_proba(rf_proba, gb_proba, rf_weight)
    soft_proba = _temperature_scale(raw_proba, temperature)

    rf_lev = rf_reg.predict(X)
    gb_lev = gb_reg.predict(X)
    soft_lev = np.clip(_ensemble_reg(rf_lev, gb_lev, rf_weight), 1.0, 100.0)

    out = df.copy()
    for i, c in enumerate(class_values):
        col = {-1: "teacher_soft_p_short", 0: "teacher_soft_p_flat", 1: "teacher_soft_p_long"}.get(int(c), f"teacher_soft_{c}")
        out[col] = soft_proba[:, i].round(6)
    out["teacher_signal"] = class_values[np.argmax(raw_proba, axis=1)]
    out["teacher_leverage"] = soft_lev.round(4)
    out["teacher_confidence"] = np.max(raw_proba, axis=1).round(6)
    return out


# ── 檢查 Teacher 是否存在 ──────────────────────────────────────────────────
def teacher_exists(model_dir: Path, symbol: str, interval: str) -> bool:
    tag = f"{symbol}_{interval}"
    t_dir = model_dir / "teacher" / tag
    return (t_dir / "teacher_meta.json").exists()


def load_teacher_report(output_dir: Path, symbol: str, interval: str) -> dict:
    tag = f"{symbol}_{interval}"
    p = output_dir / f"teacher_report_{tag}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_teacher_soft_labels(output_dir: Path, symbol: str, interval: str) -> pd.DataFrame:
    tag = f"{symbol}_{interval}"
    p = output_dir / f"teacher_soft_labels_{tag}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)
