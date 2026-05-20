from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ORDER_HISTORY_FILE = "okx_orders_history.csv"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def append_okx_order_record(
    outputs_dir: Path,
    source: str,
    symbol: str,
    interval: str,
    trade_res: dict[str, Any],
    control_payload: dict[str, Any] | None = None,
) -> Path:
    """
    Append one durable order/event record to outputs/okx_orders_history.csv.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    path = outputs_dir / ORDER_HISTORY_FILE

    decision = dict(trade_res.get("decision") or {})
    row = {
        "logged_at_utc": _utc_now_iso(),
        "source": str(source or ""),
        "symbol": str(symbol or ""),
        "interval": str(interval or ""),
        "decision_timestamp": str(decision.get("timestamp", "")),
        "signal": _safe_int(decision.get("signal", 0), 0),
        "p_long": _safe_float(decision.get("p_long", 0.0), 0.0),
        "p_short": _safe_float(decision.get("p_short", 0.0), 0.0),
        "suggested_leverage": _safe_float(decision.get("suggested_leverage", 1.0), 1.0),
        "action": str(trade_res.get("action", "")),
        "price": _safe_float(trade_res.get("price", 0.0), 0.0),
        "leverage": _safe_float(trade_res.get("leverage", 0.0), 0.0),
        "size": str(trade_res.get("size", "")),
        "simulated": bool(trade_res.get("simulated", True)),
        "enable_trading": bool(trade_res.get("enable_trading", False)),
        "inst_id": str(trade_res.get("instId", "")),
        "order_response_json": json.dumps(trade_res.get("order_response", {}), ensure_ascii=False),
        "set_leverage_response_json": json.dumps(trade_res.get("set_leverage_response", {}), ensure_ascii=False),
        "control_json": json.dumps(control_payload or {}, ensure_ascii=False),
    }

    file_exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return path


def load_okx_order_history(outputs_dir: Path, since_utc: pd.Timestamp | None = None) -> pd.DataFrame:
    path = outputs_dir / ORDER_HISTORY_FILE
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "logged_at_utc" in df.columns:
        df["logged_at_utc"] = pd.to_datetime(df["logged_at_utc"], utc=True, errors="coerce")
        if since_utc is not None:
            _since = pd.to_datetime(since_utc, utc=True, errors="coerce")
            if not pd.isna(_since):
                df = df[df["logged_at_utc"] >= _since].copy()
    return df
