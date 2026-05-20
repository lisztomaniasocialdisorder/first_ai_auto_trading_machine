from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.data_sources import interval_to_seconds
from src.paper_trade_okx import execute_latest_signal_okx
from src.pipeline import run_quick_update
from src.trade_journal import append_okx_order_record


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _report_path(outputs_dir: Path, symbol: str, interval: str) -> Path:
    tagged = outputs_dir / f"report_{symbol}_{interval}.json"
    return tagged if tagged.exists() else (outputs_dir / "report.json")


def _load_latest_decision(outputs_dir: Path, symbol: str, interval: str) -> dict[str, Any]:
    path = _report_path(outputs_dir, symbol, interval)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return dict(data.get("latest_decision") or {})


def _parse_ts(ts_value: str | None) -> datetime | None:
    if not ts_value:
        return None
    s = ts_value.strip()
    if not s:
        return None
    # Accept "Z" format and "+00:00" format.
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _state_path(outputs_dir: Path, symbol: str, interval: str) -> Path:
    return outputs_dir / f"auto_trade_state_{symbol}_{interval}.json"


def _control_path(outputs_dir: Path) -> Path:
    return outputs_dir / "auto_trade_control.json"


def _runner_status_path(outputs_dir: Path) -> Path:
    return outputs_dir / "auto_trade_runner_status.json"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def _decision_signature(decision: dict[str, Any]) -> str:
    ts = str(decision.get("timestamp", ""))
    sig = int(decision.get("signal", 0))
    lev = float(decision.get("suggested_leverage", 1.0))
    return f"{ts}|{sig}|{lev:.2f}"


def _read_control(outputs_dir: Path, default_symbol: str, default_interval: str) -> dict[str, Any]:
    default = {
        "enabled": False,
        "symbol": default_symbol,
        "interval": default_interval,
        "okx_inst_id": "BTC-USDT-SWAP",
        "okx_notional_usdt": 50.0,
        "okx_black_swan_reserve_usdt": 0.0,
        "okx_black_swan_threshold": 1.0,
        "okx_enable_trading": False,
        "okx_simulated": True,
        "okx_max_leverage": 100,
        "poll_sec": 20,
        "post_close_delay_sec": 8,
    }
    p = _control_path(outputs_dir)
    if not p.exists():
        return default
    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return default
    out = dict(default)
    out.update(raw if isinstance(raw, dict) else {})
    return out


def _save_runner_status(outputs_dir: Path, payload: dict[str, Any]) -> None:
    p = _runner_status_path(outputs_dir)
    old = {}
    if p.exists():
        try:
            old = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            old = {}
    merged = dict(old if isinstance(old, dict) else {})
    merged.update(payload)
    merged["heartbeat_utc"] = _utc_now_str()
    p.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def _apply_okx_env(control: dict[str, Any]) -> None:
    os.environ["OKX_INST_ID"] = str(control.get("okx_inst_id", "BTC-USDT-SWAP"))
    os.environ["OKX_NOTIONAL_USDT"] = str(float(control.get("okx_notional_usdt", 50.0)))
    os.environ["OKX_BLACK_SWAN_RESERVE_USDT"] = str(float(control.get("okx_black_swan_reserve_usdt", 0.0)))
    os.environ["OKX_BLACK_SWAN_THRESHOLD"] = str(float(control.get("okx_black_swan_threshold", 1.0)))
    os.environ["OKX_ENABLE_TRADING"] = "1" if bool(control.get("okx_enable_trading", False)) else "0"
    os.environ["OKX_SIMULATED"] = "1" if bool(control.get("okx_simulated", True)) else "0"
    os.environ["OKX_MAX_LEVERAGE"] = str(int(control.get("okx_max_leverage", 100)))


def run_loop(
    symbol: str,
    interval: str,
    outputs_dir: Path,
    poll_sec: int,
    post_close_delay_sec: int,
    run_once: bool,
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)

    next_run_at = time.time()
    print(f"[{_utc_now_str()}] Auto trading runner started (dashboard-controlled mode)")
    print(f"[{_utc_now_str()}] default_symbol={symbol} default_interval={interval} outputs={outputs_dir}")

    while True:
        now = time.time()
        if now < next_run_at:
            time.sleep(min(1.0, next_run_at - now))
            continue

        try:
            control = _read_control(outputs_dir, symbol, interval)
            enabled = bool(control.get("enabled", False))
            active_symbol = str(control.get("symbol", symbol) or symbol)
            active_interval = str(control.get("interval", interval) or interval)
            poll_sec_active = max(3, int(control.get("poll_sec", poll_sec)))
            post_close_delay_sec_active = max(0, int(control.get("post_close_delay_sec", post_close_delay_sec)))

            _save_runner_status(
                outputs_dir,
                {
                    "enabled": enabled,
                    "symbol": active_symbol,
                    "interval": active_interval,
                    "okx_enable_trading": bool(control.get("okx_enable_trading", False)),
                    "message": "idle (disabled by dashboard)" if not enabled else "running",
                },
            )

            if not enabled:
                next_run_at = time.time() + min(5, poll_sec_active)
                if run_once:
                    break
                continue

            _apply_okx_env(control)

            try:
                interval_sec = int(interval_to_seconds(active_interval))
            except Exception:
                interval_sec = 3600

            print(f"[{_utc_now_str()}] Quick update... ({active_symbol} {active_interval})")
            run_quick_update(symbol=active_symbol, interval=active_interval)

            decision = _load_latest_decision(outputs_dir, active_symbol, active_interval)
            if not decision:
                print(f"[{_utc_now_str()}] No latest decision found. Skip this cycle.")
                _save_runner_status(outputs_dir, {"message": "no latest decision"})
                next_run_at = time.time() + max(5, poll_sec_active)
                if run_once:
                    break
                continue

            state_file = _state_path(outputs_dir, active_symbol, active_interval)
            state = _load_state(state_file)
            last_sig = str(state.get("last_signal_signature", ""))
            sig = _decision_signature(decision)
            if sig == last_sig:
                print(f"[{_utc_now_str()}] Signal unchanged. Skip order. sig={sig}")
                _save_runner_status(outputs_dir, {"message": "signal unchanged", "last_signal_signature": sig})
            else:
                print(f"[{_utc_now_str()}] New signal detected. Execute AUTO. sig={sig}")
                trade_res = execute_latest_signal_okx(outputs_dir, active_symbol, active_interval, action_override="AUTO")
                print(
                    f"[{_utc_now_str()}] Action={trade_res.get('action')} "
                    f"price={trade_res.get('price')} lev={trade_res.get('leverage')} size={trade_res.get('size')}"
                )
                append_okx_order_record(
                    outputs_dir=outputs_dir,
                    source="background_runner",
                    symbol=active_symbol,
                    interval=active_interval,
                    trade_res=trade_res,
                    control_payload=control,
                )
                _save_state(
                    state_file,
                    {
                        "last_signal_signature": sig,
                        "last_updated_utc": _utc_now_str(),
                    },
                )
                _save_runner_status(
                    outputs_dir,
                    {
                        "message": f"executed {trade_res.get('action')}",
                        "last_signal_signature": sig,
                        "last_action": str(trade_res.get("action", "")),
                        "last_decision_timestamp": str(decision.get("timestamp", "")),
                    },
                )

            latest_ts = _parse_ts(str(decision.get("timestamp", "")))
            if latest_ts is not None:
                target = latest_ts.timestamp() + interval_sec + post_close_delay_sec_active
                next_run_at = target if target > time.time() else (time.time() + max(5, poll_sec_active))
                wait_s = int(max(0, next_run_at - time.time()))
                print(f"[{_utc_now_str()}] Next cycle in ~{wait_s}s")
            else:
                next_run_at = time.time() + max(5, poll_sec_active)

            if run_once:
                break
        except KeyboardInterrupt:
            print(f"[{_utc_now_str()}] Interrupted by user. Exiting.")
            break
        except Exception as e:  # noqa: BLE001
            print(f"[{_utc_now_str()}] Cycle failed: {e}")
            traceback.print_exc()
            _save_runner_status(outputs_dir, {"message": f"cycle failed: {e}"})
            next_run_at = time.time() + max(10, poll_sec)
            if run_once:
                break


def main() -> None:
    root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=root / ".env")

    parser = argparse.ArgumentParser(description="24/7 AI auto trading runner (quick update + OKX execution).")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h", help="5m/15m/30m/1h/1d")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--poll-sec", type=int, default=20)
    parser.add_argument("--post-close-delay-sec", type=int, default=8)
    parser.add_argument("--run-once", action="store_true", help="Run one cycle and exit.")
    args = parser.parse_args()

    run_loop(
        symbol=args.symbol,
        interval=args.interval,
        outputs_dir=Path(args.outputs),
        poll_sec=max(3, int(args.poll_sec)),
        post_close_delay_sec=max(0, int(args.post_close_delay_sec)),
        run_once=bool(args.run_once),
    )


if __name__ == "__main__":
    main()

