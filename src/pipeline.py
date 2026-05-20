from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .backtest import extract_trades, run_backtest
from .config import Settings
from .data_sources import load_or_update_ohlcv, merge_event_features
from .data_sources import interval_to_seconds
from .features import add_technical_features, build_labels
from .modeling import infer_signals, load_models, save_models, train_models


def _resolve_keep_rows() -> int:
    try:
        return max(0, int(os.getenv("KLINE_KEEP_ROWS", "0")))
    except Exception:
        return 0


def _resolve_start_ms(settings: Settings, fallback_start_ms: int) -> int:
    keep_rows = _resolve_keep_rows()
    if keep_rows <= 0:
        return int(fallback_start_ms)
    try:
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        interval_ms = int(interval_to_seconds(settings.interval) * 1000)
        if interval_ms <= 0:
            return int(fallback_start_ms)
        overlap_rows = max(200, int((settings.hours_lookback_overlap * 3600 * 1000) / interval_ms))
        need_rows = int(keep_rows + overlap_rows)
        return max(0, int(now_ms - need_rows * interval_ms))
    except Exception:
        return int(fallback_start_ms)


def _limit_recent_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    keep_rows = _resolve_keep_rows()
    if keep_rows <= 0 or df.empty:
        return df
    return df.sort_values("timestamp").tail(keep_rows).reset_index(drop=True)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_model_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def _promotion_gate(
    candidate_report: dict,
    current_report: dict | None,
    settings: Settings,
) -> tuple[bool, list[str]]:
    if not current_report:
        return True, ["no existing production model"]
    baseline = current_report.get("backtest_report") if isinstance(current_report.get("backtest_report"), dict) else current_report
    if not isinstance(baseline, dict):
        return True, ["invalid production report; promoting candidate"]

    reasons: list[str] = []

    cand_trades = int(candidate_report.get("trades") or 0)
    curr_trades = int(baseline.get("trades") or 0)
    if cand_trades < max(1, int(settings.promote_min_trades)):
        reasons.append(f"candidate trades too low ({cand_trades} < {settings.promote_min_trades})")
    if curr_trades > 0 and cand_trades < curr_trades:
        reasons.append(f"candidate has fewer trades than production ({cand_trades} < {curr_trades})")

    def _num(report: dict, key: str, default: float = 0.0) -> float:
        try:
            value = report.get(key)
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    cand_win = _num(candidate_report, "win_rate")
    curr_win = _num(baseline, "win_rate")
    if cand_win < curr_win + float(settings.promote_min_win_rate_delta):
        reasons.append(
            f"win_rate not improved enough ({cand_win:.4f} < {curr_win + float(settings.promote_min_win_rate_delta):.4f})"
        )

    cand_return = _num(candidate_report, "total_return")
    curr_return = _num(baseline, "total_return")
    if cand_return < curr_return + float(settings.promote_min_total_return_delta):
        reasons.append(
            f"total_return not improved enough ({cand_return:.4f} < {curr_return + float(settings.promote_min_total_return_delta):.4f})"
        )

    cand_dd = abs(_num(candidate_report, "max_drawdown"))
    curr_dd = abs(_num(baseline, "max_drawdown"))
    if cand_dd > curr_dd + float(settings.promote_max_drawdown_increase):
        reasons.append(
            f"max_drawdown worse than allowed ({cand_dd:.4f} > {curr_dd + float(settings.promote_max_drawdown_increase):.4f})"
        )

    cand_pf_raw = candidate_report.get("profit_factor")
    curr_pf_raw = baseline.get("profit_factor")
    if cand_pf_raw is not None and curr_pf_raw is not None:
        try:
            cand_pf = float(cand_pf_raw)
            curr_pf = float(curr_pf_raw)
            if cand_pf < curr_pf + float(settings.promote_min_profit_factor_delta):
                reasons.append(
                    f"profit_factor not improved enough ({cand_pf:.4f} < {curr_pf + float(settings.promote_min_profit_factor_delta):.4f})"
                )
        except Exception:
            pass

    return (len(reasons) == 0), reasons


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
    fallback_start_ms = int(datetime(2017, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    start_ms = _resolve_start_ms(settings, fallback_start_ms)

    fetch_settings = settings

    ohlcv = load_or_update_ohlcv(fetch_settings, start_ms=start_ms, force_full_refresh=force_full_refresh)
    ohlcv = _limit_recent_ohlcv(ohlcv)
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
        progress_cb(65, "標籤建構完成，準備開始訓練模型（可能需要幾分鐘）")

    # ── 自動偵測 Teacher 軟標籤，若存在則將蒸餾混入訓練 ─────────
    tag = f"{settings.symbol}_{settings.interval}"
    soft_label_path = settings.output_dir / f"teacher_soft_labels_{tag}.csv"
    soft_labels_df: pd.DataFrame | None = None
    distill_alpha = 0.4
    if soft_label_path.exists():
        try:
            soft_labels_df = pd.read_csv(soft_label_path)
            if progress_cb:
                progress_cb(66, f"偵測到 Teacher 軟標籤（{len(soft_labels_df):,} 筆），啟用蒸餾訓練（alpha={distill_alpha}）")
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
    candidate_model_dir = settings.model_dir / "_staged" / tag / datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    production_model_dir = settings.model_dir / tag
    save_models(models, candidate_model_dir)

    inferred = infer_signals(labeled_full, models, settings)
    bt_curve, bt_report = run_backtest(inferred, settings)
    if progress_cb:
        progress_cb(95, "回測完成，寫入輸出檔")

    production_report_path = settings.output_dir / f"production_report_{tag}.json"
    previous_production_report = _read_json(production_report_path)
    promote, promote_reasons = _promotion_gate(bt_report, previous_production_report, settings)
    promoted = False
    if promote:
        _copy_model_tree(candidate_model_dir, production_model_dir)
        promoted = True
        _write_json(
            production_report_path,
            {
                "tag": tag,
                "promoted_at_utc": datetime.now(tz=timezone.utc).isoformat(),
                "candidate_model_dir": str(candidate_model_dir),
                "production_model_dir": str(production_model_dir),
                "backtest_report": bt_report,
                "train_metrics": train_metrics,
            },
        )

    promotion_status = {
        "tag": tag,
        "candidate_model_dir": str(candidate_model_dir),
        "production_model_dir": str(production_model_dir),
        "promoted": promoted,
        "reasons": promote_reasons if not promoted else ["candidate passed promotion gate"],
        "candidate_backtest_report": bt_report,
        "previous_production_report": previous_production_report,
        "train_metrics": train_metrics,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    _write_json(settings.output_dir / f"model_promotion_{tag}.json", promotion_status)

    out = _write_outputs(inferred, bt_curve, train_metrics, bt_report, settings)
    out["model_promotion"] = promotion_status
    _write_json(settings.output_dir / f"report_{tag}.json", out)
    _write_json(settings.output_dir / "report.json", out)
    if progress_cb:
        progress_cb(100, "全部完成")
    return out


def run_quick_update(symbol: str | None = None, interval: str | None = None) -> dict:
    settings = Settings(symbol=symbol, interval=interval)
    fallback_start_ms = int(datetime(2017, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    start_ms = _resolve_start_ms(settings, fallback_start_ms)

    fetch_settings = settings

    ohlcv = load_or_update_ohlcv(fetch_settings, start_ms=start_ms, force_full_refresh=False)
    ohlcv = _limit_recent_ohlcv(ohlcv)
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
