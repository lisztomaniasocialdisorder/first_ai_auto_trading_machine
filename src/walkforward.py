from __future__ import annotations

from dataclasses import replace
from datetime import timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import Settings
from .data_sources import interval_to_seconds
from .modeling import infer_signals, train_models


def _expectancy_unit(win_rate: float, pnl_ratio: float) -> float:
    w = max(0.0, min(1.0, float(win_rate)))
    r = max(0.0, float(pnl_ratio))
    return (w * r) - (1.0 - w)


def _wf_min_train_for_interval(interval: str) -> int:
    return {
        "5m": 1400,
        "15m": 1100,
        "30m": 900,
        "1h": 700,
        "1d": 240,
    }.get(str(interval), 700)


def _wf_min_test_for_interval(interval: str) -> int:
    return {
        "5m": 140,
        "15m": 120,
        "30m": 100,
        "1h": 80,
        "1d": 40,
    }.get(str(interval), 80)


def run_walkforward_validation(
    df: pd.DataFrame,
    settings: Settings,
    *,
    n_folds: int = 4,
    min_train_rows: int | None = None,
    test_rows: int | None = None,
) -> dict:
    x = df.copy().sort_values("timestamp").reset_index(drop=True)
    if "timestamp" in x.columns:
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    x = x.dropna(subset=["timestamp", "close", "label"]).reset_index(drop=True)

    total_rows = int(len(x))
    interval_sec = max(1, int(interval_to_seconds(settings.interval)))
    embargo_bars = max(1, int(np.ceil((float(settings.future_horizon_hours) * 3600.0) / float(interval_sec))))

    base_min_train = int(_wf_min_train_for_interval(settings.interval))
    req_min_train = int(base_min_train if min_train_rows is None else min_train_rows)
    req_min_train = max(80, req_min_train)
    min_train = req_min_train
    # 若樣本不足，對低週期/高週期都採自適應縮放，至少能跑出可檢視的 OOS 結果。
    if total_rows < (min_train + max(40, 2 * embargo_bars + 20)):
        min_train = max(80, int(total_rows * 0.55))
    min_train = min(min_train, max(80, total_rows - max(40, 2 * embargo_bars + 20)))
    if total_rows < (min_train + max(40, 2 * embargo_bars + 20)):
        raise RuntimeError(f"walk-forward 資料不足，需要至少 {min_train + max(40, 2 * embargo_bars + 20)} 根，目前只有 {total_rows} 根")

    min_test = int(_wf_min_test_for_interval(settings.interval))
    default_test_rows = max(min_test, int(total_rows * 0.10))
    fold_test_rows = int(test_rows or default_test_rows)
    fold_test_rows = max(min_test, min(fold_test_rows, max(min_test, total_rows // max(2, n_folds + 1))))

    # 樣本不足時自動降低 fold 數，避免直接報錯卡住。
    max_folds_possible = max(1, int((total_rows - min_train - embargo_bars) // max(1, fold_test_rows + embargo_bars)))
    n_folds = max(1, min(int(n_folds), max_folds_possible))

    wf_settings = replace(settings, max_train_rows=0, min_train_rows=int(min_train))
    train_end = min_train
    folds: list[dict] = []
    fold_curves: list[pd.DataFrame] = []

    for fold_idx in range(1, n_folds + 1):
        train_stop = max(min_train, train_end - embargo_bars)
        test_start = min(total_rows, train_end + embargo_bars)
        test_end = test_start + fold_test_rows
        if test_end > total_rows:
            break

        train_df = x.iloc[:train_stop].dropna().reset_index(drop=True)
        test_df = x.iloc[test_start:test_end].copy().reset_index(drop=True)
        if len(train_df) < wf_settings.min_train_rows or test_df.empty:
            break

        models, train_metrics = train_models(train_df, wf_settings, progress_cb=None, soft_labels_df=None, distill_alpha=0.0)
        inferred = infer_signals(test_df, models, wf_settings)
        bt_curve, bt_report = run_backtest(inferred, wf_settings, interval=wf_settings.interval)
        fold_curves.append(bt_curve.assign(fold=fold_idx))

        folds.append(
            {
                "fold": fold_idx,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "embargo_bars": int(embargo_bars),
                "train_start_utc": str(train_df["timestamp"].iloc[0]),
                "train_end_utc": str(train_df["timestamp"].iloc[-1]),
                "test_start_utc": str(test_df["timestamp"].iloc[0]),
                "test_end_utc": str(test_df["timestamp"].iloc[-1]),
                "backtest_report": bt_report,
                "train_metrics": {
                    "train_rows": train_metrics.get("train_rows"),
                    "test_rows": train_metrics.get("test_rows"),
                    "classification_report": train_metrics.get("classification_report"),
                },
            }
        )
        train_end = test_end

    if not folds:
        raise RuntimeError("walk-forward 無法建立任何有效 fold")

    fold_returns = [float(f["backtest_report"].get("total_return") or 0.0) for f in folds]
    fold_drawdowns = [abs(float(f["backtest_report"].get("max_drawdown") or 0.0)) for f in folds]
    fold_sharpes = [float(f["backtest_report"].get("sharpe") or 0.0) for f in folds]
    fold_win_rates = [float(f["backtest_report"].get("win_rate") or 0.0) for f in folds]
    fold_pnl_ratios = [float(f["backtest_report"].get("pnl_ratio") or 0.0) for f in folds]
    fold_expectancies = [_expectancy_unit(w, r) for w, r in zip(fold_win_rates, fold_pnl_ratios)]
    fold_trades = [int(f["backtest_report"].get("trades") or 0) for f in folds]

    compounded_return = float(np.prod([1.0 + r for r in fold_returns]) - 1.0)
    full_start = str(x["timestamp"].iloc[0])
    full_end = str(x["timestamp"].iloc[-1])

    return {
        "symbol": settings.symbol,
        "interval": settings.interval,
        "generated_at_utc": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "source_rows": total_rows,
        "source_start_utc": full_start,
        "source_end_utc": full_end,
        "fold_count": len(folds),
        "test_rows_per_fold": fold_test_rows,
        "embargo_bars": int(embargo_bars),
        "summary": {
            "compounded_total_return": compounded_return,
            "average_fold_return": float(np.mean(fold_returns)),
            "median_fold_return": float(np.median(fold_returns)),
            "average_fold_win_rate": float(np.mean(fold_win_rates)),
            "average_fold_pnl_ratio": float(np.mean(fold_pnl_ratios)) if fold_pnl_ratios else 0.0,
            "average_fold_expectancy_unit": float(np.mean(fold_expectancies)) if fold_expectancies else 0.0,
            "average_fold_sharpe": float(np.mean(fold_sharpes)),
            "worst_fold_drawdown": float(max(fold_drawdowns)) if fold_drawdowns else 0.0,
            "total_fold_trades": int(sum(fold_trades)),
            "average_fold_trades": float(np.mean(fold_trades)) if fold_trades else 0.0,
            "positive_folds": int(sum(1 for r in fold_returns if r > 0)),
            "positive_expectancy_folds": int(sum(1 for e in fold_expectancies if e > 0)),
        },
        "folds": folds,
    }


def save_walkforward_report(report: dict, output_dir: Path, tag: str) -> Path:
    path = output_dir / f"walkforward_report_{tag}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
