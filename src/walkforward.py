from __future__ import annotations

from dataclasses import replace
from datetime import timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import Settings
from .modeling import infer_signals, train_models


def _expectancy_unit(win_rate: float, pnl_ratio: float) -> float:
    w = max(0.0, min(1.0, float(win_rate)))
    r = max(0.0, float(pnl_ratio))
    return (w * r) - (1.0 - w)


def run_walkforward_validation(
    df: pd.DataFrame,
    settings: Settings,
    *,
    n_folds: int = 4,
    min_train_rows: int = 1000,
    test_rows: int | None = None,
) -> dict:
    x = df.copy().sort_values("timestamp").reset_index(drop=True)
    if "timestamp" in x.columns:
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    x = x.dropna(subset=["timestamp", "close", "label"]).reset_index(drop=True)

    total_rows = int(len(x))
    min_train = max(int(settings.min_train_rows), int(min_train_rows))
    if total_rows < (min_train + 200):
        raise RuntimeError(f"walk-forward 資料不足，需要至少 {min_train + 200} 根，目前只有 {total_rows} 根")

    default_test_rows = max(150, int(total_rows * 0.10))
    fold_test_rows = int(test_rows or default_test_rows)
    fold_test_rows = max(100, min(fold_test_rows, max(100, total_rows // max(2, n_folds + 1))))

    wf_settings = replace(settings, max_train_rows=0)
    train_end = min_train
    folds: list[dict] = []
    fold_curves: list[pd.DataFrame] = []

    for fold_idx in range(1, n_folds + 1):
        test_end = train_end + fold_test_rows
        if test_end > total_rows:
            break

        train_df = x.iloc[:train_end].dropna().reset_index(drop=True)
        test_df = x.iloc[train_end:test_end].copy().reset_index(drop=True)
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
