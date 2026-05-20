from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from .backtest import extract_trades, run_backtest
from .config import Settings
from .data_sources import load_or_update_ohlcv, merge_event_features
from .data_sources import interval_to_seconds
from .features import add_technical_features, build_labels
from .modeling import infer_signals, load_models, save_models, train_models



def _write_outputs(
    inferred: pd.DataFrame,
    bt_curve: pd.DataFrame,
    train_metrics: dict,
    bt_report: dict,
    settings: Settings,
) -> dict:
    latest = bt_curve.iloc[-1]
    decision = {
        "timestamp": str(latest["timestamp"]),
        "price": float(latest["close"]),
        "signal": int(latest["signal"]),
        "suggested_leverage": float(latest["suggested_leverage"]),
        "max_safe_leverage": float(latest["max_safe_leverage"]),
        "p_long": float(latest["p_long"]),
        "p_short": float(latest["p_short"]),
    }

    results = {
        "train_metrics": train_metrics,
        "backtest_report": bt_report,
        "latest_decision": decision,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }

    tag = f"{settings.symbol}_{settings.interval}"
    bt_curve.to_csv(settings.output_dir / f"backtest_curve_{tag}.csv", index=False)
    inferred.to_csv(settings.output_dir / f"signals_with_features_{tag}.csv", index=False)
    with open(settings.output_dir / f"report_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    trades = extract_trades(bt_curve)
    trades.to_csv(settings.output_dir / f"trades_{tag}.csv", index=False, encoding="utf-8")

    # Backward-compat: keep updating the legacy filenames to the latest run,
    # so older UI/exe that expects fixed paths still works.
    bt_curve.to_csv(settings.output_dir / "backtest_curve.csv", index=False)
    inferred.to_csv(settings.output_dir / "signals_with_features.csv", index=False)
    with open(settings.output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    trades.to_csv(settings.output_dir / "trades.csv", index=False, encoding="utf-8")
    return results


def run_pipeline(
    force_full_refresh: bool = False,
    progress_cb: Callable[[int, str], None] | None = None,
    symbol: str | None = None,
    interval: str | None = None,
) -> dict:
    settings = Settings(symbol=symbol, interval=interval)
    if progress_cb:
        progress_cb(5, "初始化設定")

    # BTC spot data starts long ago; Jan 1, 2017 UTC is a practical baseline.
    start_ms = int(datetime(2017, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    fetch_settings = settings

    ohlcv = load_or_update_ohlcv(fetch_settings, start_ms=start_ms, force_full_refresh=force_full_refresh)
    if progress_cb:
        progress_cb(25, "市場資料更新完成")
    with_events = merge_event_features(ohlcv, settings)
    if progress_cb:
        progress_cb(40, "事件與情緒特徵合併完成")
    feat = add_technical_features(with_events)
    if progress_cb:
        progress_cb(55, "技術指標特徵完成")

    bars = int(round((settings.future_horizon_hours * 3600) / interval_to_seconds(settings.interval)))
    labeled_full = build_labels(feat, horizon_bars=bars, long_th=settings.long_threshold, short_th=settings.short_threshold)
    labeled_train = labeled_full.dropna().reset_index(drop=True)
    if progress_cb:
        progress_cb(65, "標籤建構完成，準備開始訓練模型 (可能需要幾分鐘，請耐心等候)...")

    # ── 自動偵測 Teacher 軟標籤，若存在則將蒸餾混入訓練 ─────────
    tag = f"{settings.symbol}_{settings.interval}"
    soft_label_path = settings.output_dir / f"teacher_soft_labels_{tag}.csv"
    soft_labels_df: pd.DataFrame | None = None
    distill_alpha = 0.4
    if soft_label_path.exists():
        try:
            soft_labels_df = pd.read_csv(soft_label_path)
            if progress_cb:
                progress_cb(66, f"偵測到 Teacher 軟標籤（{len(soft_labels_df):,} 筆），開啟蒸餾模式（alpha={distill_alpha}）...")
        except Exception as _e:
            import warnings
            warnings.warn(f"[pipeline] 無法載入 Teacher 軟標籤：{_e}，退回純硬標籤訓練。", stacklevel=2)
            soft_labels_df = None

    models, train_metrics = train_models(
        labeled_train, settings, progress_cb,
        soft_labels_df=soft_labels_df,
        distill_alpha=distill_alpha,
    )
    if progress_cb:
        progress_cb(80, "模型訓練完成")
    model_dir = settings.model_dir / f"{settings.symbol}_{settings.interval}"
    save_models(models, model_dir)

    inferred = infer_signals(labeled_full, models, settings)
    bt_curve, bt_report = run_backtest(inferred, settings)
    if progress_cb:
        progress_cb(95, "回測完成，寫入輸出檔")

    out = _write_outputs(inferred, bt_curve, train_metrics, bt_report, settings)
    if progress_cb:
        progress_cb(100, "全部完成")
    return out


def run_quick_update(symbol: str | None = None, interval: str | None = None) -> dict:
    settings = Settings(symbol=symbol, interval=interval)
    start_ms = int(datetime(2017, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    fetch_settings = settings

    ohlcv = load_or_update_ohlcv(fetch_settings, start_ms=start_ms, force_full_refresh=False)
    # Keep quick update fast: only compute on a rolling recent window.
    bars_per_day = int(round((24 * 3600) / interval_to_seconds(settings.interval)))
    window_bars = int(max(500, settings.quick_window_days * bars_per_day))
    ohlcv = ohlcv.sort_values("timestamp").tail(window_bars).reset_index(drop=True)

    with_events = merge_event_features(ohlcv, settings, fast_mode=True)
    feat = add_technical_features(with_events)
    bars = int(round((settings.future_horizon_hours * 3600) / interval_to_seconds(settings.interval)))
    labeled_full = build_labels(feat, horizon_bars=bars, long_th=settings.long_threshold, short_th=settings.short_threshold)
    
    model_dir = settings.model_dir / f"{settings.symbol}_{settings.interval}"
    models = load_models(model_dir)
    inferred = infer_signals(labeled_full, models, settings)
    bt_curve, bt_report = run_backtest(inferred, settings)

    train_metrics = {
        "note": "quick_update used saved models (no retraining)",
        "train_rows": None,
        "test_rows": None,
        "training_backend": getattr(models, "backend", "unknown"),
        "training_device": ((getattr(models, "backend_meta", {}) or {}).get("device", "unknown")),
    }
    return _write_outputs(inferred, bt_curve, train_metrics, bt_report, settings)


def pretty_print_results(results: dict) -> None:
    print("=== Latest Decision ===")
    print(json.dumps(results["latest_decision"], indent=2, ensure_ascii=False))
    print("\n=== Backtest Report ===")
    print(json.dumps(results["backtest_report"], indent=2, ensure_ascii=False))
    print("\n=== Train Metrics (summary) ===")

    cls = results["train_metrics"]["classification_report"]
    weighted = cls.get("weighted avg", {})
    summary = {
        "weighted_precision": weighted.get("precision"),
        "weighted_recall": weighted.get("recall"),
        "weighted_f1": weighted.get("f1-score"),
        "leverage_mae": results["train_metrics"].get("leverage_mae"),
        "train_rows": results["train_metrics"].get("train_rows"),
        "test_rows": results["train_metrics"].get("test_rows"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
