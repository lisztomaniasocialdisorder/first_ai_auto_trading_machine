from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd

from .cache import JsonCache
from .exchange_okx import OKXClient, OKXCredentials


@dataclass
class OKXTradeConfig:
    inst_id: str | None = None
    inst_type: str | None = None
    td_mode: str | None = None  # isolated/cross/cash
    pos_mode: str | None = None  # net or long_short
    simulated: bool | None = None
    base_url: str | None = None
    enable_trading: bool | None = None

    # Position sizing
    notional_usdt: float | None = None
    max_leverage: int | None = None
    black_swan_reserve_usdt: float | None = None
    black_swan_threshold: float | None = None

    def __post_init__(self) -> None:
        self.inst_id = self.inst_id or os.getenv("OKX_INST_ID", "BTC-USDT-SWAP")
        self.inst_type = self.inst_type or os.getenv("OKX_INST_TYPE", "SWAP")
        self.td_mode = self.td_mode or os.getenv("OKX_TD_MODE", "isolated")
        self.pos_mode = self.pos_mode or os.getenv("OKX_POS_MODE", "net")
        self.simulated = bool(int(os.getenv("OKX_SIMULATED", "1"))) if self.simulated is None else self.simulated
        self.base_url = self.base_url or os.getenv("OKX_BASE_URL", "https://www.okx.com")
        self.enable_trading = (os.getenv("OKX_ENABLE_TRADING", "0") == "1") if self.enable_trading is None else self.enable_trading

        self.notional_usdt = float(self.notional_usdt or os.getenv("OKX_NOTIONAL_USDT", "50"))
        self.max_leverage = int(self.max_leverage or os.getenv("OKX_MAX_LEVERAGE", "100"))
        self.black_swan_reserve_usdt = float(self.black_swan_reserve_usdt or os.getenv("OKX_BLACK_SWAN_RESERVE_USDT", "0"))
        self.black_swan_threshold = float(self.black_swan_threshold or os.getenv("OKX_BLACK_SWAN_THRESHOLD", "1.0"))


def _okx_client_from_env(cfg: OKXTradeConfig) -> OKXClient:
    key = os.getenv("OKX_API_KEY", "")
    sec = os.getenv("OKX_API_SECRET", "")
    pas = os.getenv("OKX_API_PASSPHRASE", "")
    if not (key and sec and pas):
        raise RuntimeError("Missing OKX credentials. Set OKX_API_KEY/OKX_API_SECRET/OKX_API_PASSPHRASE in env.")
    return OKXClient(creds=OKXCredentials(api_key=key, secret_key=sec, passphrase=pas), base_url=cfg.base_url, simulated=cfg.simulated)


def _load_latest_decision(outputs_dir: Path, symbol: str, interval: str) -> dict[str, Any]:
    tag = f"{symbol}_{interval}"
    p = outputs_dir / f"report_{tag}.json"
    if not p.exists():
        # fallback legacy
        p = outputs_dir / "report.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("latest_decision", {})


def _load_latest_price(outputs_dir: Path, symbol: str, interval: str) -> float:
    tag = f"{symbol}_{interval}"
    p = outputs_dir / f"signals_with_features_{tag}.csv"
    if not p.exists():
        p = outputs_dir / "signals_with_features.csv"
    df = pd.read_csv(p)
    return float(df["close"].iloc[-1])


def _load_latest_signal_row(outputs_dir: Path, symbol: str, interval: str) -> dict[str, Any]:
    tag = f"{symbol}_{interval}"
    p = outputs_dir / f"signals_with_features_{tag}.csv"
    if not p.exists():
        p = outputs_dir / "signals_with_features.csv"
    df = pd.read_csv(p)
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()


def _signal_direction(sig: int) -> int:
    if sig > 0:
        return 1
    if sig < 0:
        return -1
    return 0


def _load_mtf_snapshot(outputs_dir: Path, symbol: str, intervals: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for tf in intervals:
        try:
            decision = _load_latest_decision(outputs_dir, symbol, tf)
        except Exception:
            decision = {}
        try:
            latest_row = _load_latest_signal_row(outputs_dir, symbol, tf)
        except Exception:
            latest_row = {}

        sig_raw = decision.get("signal", latest_row.get("signal", 0))
        lev_raw = decision.get("suggested_leverage", latest_row.get("suggested_leverage", 1.0))
        price_raw = decision.get("price", latest_row.get("close", 0.0))
        try:
            sig = int(sig_raw or 0)
        except Exception:
            sig = 0
        try:
            suggested_lev = float(lev_raw or 1.0)
        except Exception:
            suggested_lev = 1.0
        try:
            price = float(price_raw or 0.0)
        except Exception:
            price = 0.0

        snapshot[tf] = {
            "decision": decision,
            "row": latest_row,
            "signal": sig,
            "direction": _signal_direction(sig),
            "suggested_leverage": suggested_lev,
            "price": price,
        }
    return snapshot


def _pick_mtf_consensus(snapshot: dict[str, dict[str, Any]]) -> dict[str, Any]:
    order = ("5m", "15m", "30m", "1h", "1d")
    pairs = (
        ("5m", "15m", 1),
        ("15m", "30m", 2),
        ("30m", "1h", 3),
        ("1h", "1d", 4),
    )
    cap_by_tier = {1: 8, 2: 6, 3: 4, 4: 3}
    candidates: list[dict[str, Any]] = []

    for left, right, tier in pairs:
        dl = int((snapshot.get(left) or {}).get("direction", 0) or 0)
        dr = int((snapshot.get(right) or {}).get("direction", 0) or 0)
        if dl != 0 and dl == dr:
            candidates.append(
                {
                    "direction": dl,
                    "tier": tier,
                    "pair": (left, right),
                    "leverage_cap": cap_by_tier.get(tier, 3),
                }
            )

    if not candidates:
        return {
            "direction": 0,
            "tier": 0,
            "pair": None,
            "leverage_cap": 1,
            "note": "no adjacent timeframe consensus",
        }

    chosen = max(candidates, key=lambda item: int(item.get("tier", 0) or 0))
    pair_left, pair_right = chosen["pair"]
    direction_label = "long" if int(chosen["direction"]) > 0 else "short"
    return {
        "direction": int(chosen["direction"]),
        "tier": int(chosen["tier"]),
        "pair": chosen["pair"],
        "leverage_cap": int(chosen["leverage_cap"]),
        "note": f"mtf consensus: {pair_left}+{pair_right} => {direction_label}, leverage cap {int(chosen['leverage_cap'])}x",
    }


def _extract_usdt_balance(balance_resp: dict[str, Any]) -> float | None:
    data = balance_resp.get("data") or []
    if not data:
        return None
    details = data[0].get("details") or []
    for d in details:
        if str(d.get("ccy", "")).upper() != "USDT":
            continue
        for k in ("availBal", "eq", "cashBal"):
            try:
                return float(d.get(k) or 0)
            except Exception:
                continue
    return None


def _pick_pos_side(pos_mode: str, desired: int) -> Optional[str]:
    if pos_mode == "net":
        return None
    if desired > 0:
        return "long"
    if desired < 0:
        return "short"
    return None


def _looks_like_posside_error(exc: Exception) -> bool:
    s = str(exc)
    return ("51000" in s) and ("posSide" in s or "posside" in s)


def _position_side_and_size(pos_row: dict[str, Any]) -> tuple[Optional[Literal["long", "short"]], float]:
    """
    Normalize OKX position record to (side, abs_size).
    In net mode, posSide is usually 'net' and signed pos indicates side.
    """
    raw = float(pos_row.get("pos") or 0)
    if raw == 0:
        return None, 0.0
    side = str(pos_row.get("posSide") or "").lower()
    if side == "long":
        return "long", abs(raw)
    if side == "short":
        return "short", abs(raw)
    # net/empty: infer from signed quantity
    return ("long", abs(raw)) if raw > 0 else ("short", abs(raw))


def _close_order_args(
    cfg: "OKXTradeConfig",
    side: Literal["long", "short"],
    size: float,
) -> dict[str, Any]:
    close_side: Literal["buy", "sell"] = "sell" if side == "long" else "buy"
    close_pos_side = side if cfg.pos_mode == "long_short" else None
    return {
        "instId": cfg.inst_id,
        "tdMode": cfg.td_mode,
        "side": close_side,
        "ordType": "market",
        "sz": str(abs(size)),
        "posSide": close_pos_side,
        "reduceOnly": True,
    }


def _pos_mode_from_account_config(cfg_resp: dict[str, Any]) -> str:
    """
    OKX account/config returns posMode like 'net_mode' or 'long_short_mode' (naming can vary).
    Normalize to our config values: 'net' or 'long_short'.
    """
    data = (cfg_resp.get("data") or [])
    if not data:
        return "net"
    pos_mode = str(data[0].get("posMode") or "").lower()
    if "long" in pos_mode and "short" in pos_mode:
        return "long_short"
    if "net" in pos_mode:
        return "net"
    return "net"


def _calc_swap_contract_size(client: OKXClient, cfg: OKXTradeConfig, price: float) -> str:
    """
    Estimate swap contract size by instrument ctVal/ctValCcy if available.
    Falls back to treating sz as USDT amount (may fail depending on instrument rules).
    """
    inst = client.get_instruments(cfg.inst_type, cfg.inst_id)
    data = (inst.get("data") or [])
    if not data:
        return str(max(1, int(cfg.notional_usdt / 10)))

    row = data[0]
    ct_val = float(row.get("ctVal", "0") or 0)
    ct_ccy = row.get("ctValCcy") or ""
    lot_sz = float(row.get("lotSz", "1") or 1)

    if ct_val > 0:
        if str(ct_ccy).upper() == "USDT":
            contracts = cfg.notional_usdt / ct_val
        else:
            # Assume ctVal is in base coin (e.g., BTC); convert via price.
            contracts = cfg.notional_usdt / (ct_val * max(price, 1e-9))
    else:
        contracts = cfg.notional_usdt / max(price, 1e-9)

    # Snap to lot size
    contracts = max(lot_sz, (int(contracts / lot_sz) * lot_sz))
    return str(int(contracts)) if contracts >= 1 else str(contracts)


def execute_latest_signal_okx(
    outputs_dir: Path,
    symbol: str,
    interval: str,
    leverage_override: Optional[int] = None,
    action_override: Optional[Literal["AUTO", "LONG", "SHORT", "CLOSE"]] = None,
) -> dict[str, Any]:
    """
    Paper-trade (simulated) execution on OKX based on the latest model signal.
    Safe-by-default: requires OKX_ENABLE_TRADING=1 to actually place orders.
    """
    cfg = OKXTradeConfig()
    client = _okx_client_from_env(cfg)

    # Preflight auth check to provide clearer error messages early.
    acct_cfg = client.get_account_config()
    # If user did not set OKX_POS_MODE explicitly, auto-detect for safer defaults.
    if os.getenv("OKX_POS_MODE", "") == "":
        cfg.pos_mode = _pos_mode_from_account_config(acct_cfg)

    decision = _load_latest_decision(outputs_dir, symbol, interval)
    latest_row = _load_latest_signal_row(outputs_dir, symbol, interval)
    mtf_snapshot = _load_mtf_snapshot(outputs_dir, symbol, ("5m", "15m", "30m", "1h", "1d"))
    mtf_gate = _pick_mtf_consensus(mtf_snapshot)
    sig = int(decision.get("signal", 0))
    suggested_lev = float(decision.get("suggested_leverage", 1.0))
    price = float(decision.get("price", 0.0) or latest_row.get("close", 0.0) or 0.0) or _load_latest_price(outputs_dir, symbol, interval)
    black_swan_score = float(latest_row.get("black_swan_risk_score", 0.0) or 0.0)
    panic_news_score = float(latest_row.get("panic_news_score", 0.0) or 0.0)
    war_news_score = float(latest_row.get("war_news_score", 0.0) or 0.0)
    macro_event_risk_score = float(latest_row.get("macro_event_risk_score", 0.0) or 0.0)
    market_panic_score = float(latest_row.get("market_panic_score", 0.0) or 0.0)
    mode = action_override or "AUTO"

    effective_notional_usdt = float(cfg.notional_usdt)
    hold_due_to_black_swan = False
    risk_control_note = ""
    black_swan_active = bool(cfg.black_swan_reserve_usdt > 0 and black_swan_score >= cfg.black_swan_threshold)

    # User policy:
    # - In black swan state, AUTO mode must pause (manual decisions only).
    # - Reserve capital is a manual buffer and must not be auto-operated.
    if black_swan_active and mode == "AUTO":
        hold_due_to_black_swan = True
        risk_control_note = "hold: black swan active, auto trading paused; manual control only"

    # For AUTO mode (non-black-swan hold), cap notional by reserve policy.
    if (not hold_due_to_black_swan) and mode == "AUTO" and cfg.black_swan_reserve_usdt > 0:
        try:
            bal_resp = client.get_balance("USDT")
            usdt_balance = _extract_usdt_balance(bal_resp)
        except Exception:
            usdt_balance = None
        if usdt_balance is not None:
            effective_notional_usdt = min(float(cfg.notional_usdt), max(0.0, float(usdt_balance) - float(cfg.black_swan_reserve_usdt)))
            if effective_notional_usdt < 5.0:
                hold_due_to_black_swan = True
                risk_control_note = "hold: black swan reserve leaves notional below minimum"

    # Decide desired action
    if mode == "LONG":
        desired = 1
    elif mode == "SHORT":
        desired = -1
    elif mode == "CLOSE":
        desired = 0
    else:
        desired = int(mtf_gate.get("direction", 0) or 0)
        if desired == 0:
            risk_control_note = (risk_control_note + " | " if risk_control_note else "") + "mtf gate: no adjacent consensus"
        else:
            risk_control_note = (risk_control_note + " | " if risk_control_note else "") + str(mtf_gate.get("note", ""))

    # Auto risk regime from news/events:
    # - If panic rises (but not severe black swan hold), bias to defensive short.
    panic_regime = bool((market_panic_score >= 2.0) or (panic_news_score >= 1.0) or (war_news_score >= 1.0))
    if mode == "AUTO" and panic_regime and not hold_due_to_black_swan:
        desired = -1
        risk_control_note = (risk_control_note + " | " if risk_control_note else "") + "panic regime: prefer short"

    if hold_due_to_black_swan and mode != "CLOSE":
        desired = 0

    lever = int(leverage_override or min(cfg.max_leverage, max(1, int(round(suggested_lev)))))
    if mode == "AUTO" and desired != 0:
        lever = min(lever, int(mtf_gate.get("leverage_cap", lever) or lever))
    if mode == "AUTO":
        if macro_event_risk_score > 0:
            # CPI/PPI/FOMC release windows: auto reduce risk.
            lever = max(1, int(round(lever * 0.5)))
            effective_notional_usdt = max(5.0, float(effective_notional_usdt) * 0.6)
            risk_control_note = (risk_control_note + " | " if risk_control_note else "") + "macro release window: reduced leverage/notional"
        if panic_regime:
            lever = max(1, min(lever, 3))
            effective_notional_usdt = max(5.0, float(effective_notional_usdt) * 0.7)
            risk_control_note = (risk_control_note + " | " if risk_control_note else "") + "panic: leverage capped"

    pos_side = _pick_pos_side(cfg.pos_mode, desired)

    # Set leverage for swap/futures modes
    lev_resp = None
    if cfg.inst_type in ("SWAP", "FUTURES") and desired != 0:
        # Avoid spamming set-leverage: cache last successful setting per instId/posSide/mgnMode.
        cache_path = outputs_dir / "okx_leverage_cache.json"
        cache = JsonCache(cache_path)
        cache_key = f"{cfg.inst_id}:{cfg.td_mode}:{pos_side or 'net'}"
        cached = (cache.read() or {}).get(cache_key, {})
        cached_lev = int(cached.get("lever", 0) or 0)
        if cached_lev == int(lever):
            lev_resp = {
                "cached": True,
                "instId": cfg.inst_id,
                "lever": str(lever),
                "mgnMode": cfg.td_mode,
                "posSide": (pos_side or ""),
            }
        else:
            try:
                lev_resp = client.set_leverage(cfg.inst_id, lever=lever, mgn_mode=cfg.td_mode, pos_side=pos_side)
            except Exception as e:  # noqa: BLE001
                # Some accounts require posSide when in long/short mode, OKX returns code 51000.
                if _looks_like_posside_error(e):
                    forced_pos_side = "long" if desired > 0 else "short"
                    lev_resp = client.set_leverage(cfg.inst_id, lever=lever, mgn_mode=cfg.td_mode, pos_side=forced_pos_side)
                    pos_side = forced_pos_side
                else:
                    raise

            # Save successful leverage setting in cache (best-effort).
            try:
                payload = cache.read() or {}
                new_key = f"{cfg.inst_id}:{cfg.td_mode}:{pos_side or 'net'}"
                payload[new_key] = {"lever": int(lever)}
                cache.write(payload)
            except Exception:
                pass

    original_notional = float(cfg.notional_usdt)
    cfg.notional_usdt = float(effective_notional_usdt)
    sz = _calc_swap_contract_size(client, cfg, price) if cfg.inst_type in ("SWAP", "FUTURES") else str(cfg.notional_usdt)
    cfg.notional_usdt = original_notional

    action: Literal["HOLD", "OPEN_LONG", "OPEN_SHORT", "CLOSE"] = "HOLD"
    order_resp = None

    if mode == "CLOSE":
        action = "CLOSE"
        # Close any existing positions for this instrument.
        pos = client.get_positions(inst_type=cfg.inst_type, inst_id=cfg.inst_id)
        pdata = pos.get("data") or []
        close_orders = []
        for p in pdata:
            try:
                p_inst = p.get("instId")
                if p_inst != cfg.inst_id:
                    continue
                p_side, p_size = _position_side_and_size(p)
                if p_side is None or p_size <= 0:
                    continue
                args = _close_order_args(cfg, p_side, p_size)

                if not cfg.enable_trading:
                    close_orders.append({"dry_run": True, **args})
                else:
                    close_orders.append(
                        client.place_order(
                            inst_id=args["instId"],
                            td_mode=args["tdMode"],
                            side=args["side"],
                            ord_type=args["ordType"],
                            sz=args["sz"],
                            pos_side=args["posSide"],
                            reduce_only=True,
                            lever=None,
                        )
                    )
            except Exception:
                continue
        order_resp = {"close_orders": close_orders, "positions": pdata}
        if not close_orders:
            order_resp = {"note": "no open positions found", "positions": pdata}
    elif desired == 0:
        action = "HOLD"
        if risk_control_note:
            order_resp = {"note": risk_control_note}
    else:
        action = "OPEN_LONG" if desired > 0 else "OPEN_SHORT"
        side: Literal["buy", "sell"] = "buy" if desired > 0 else "sell"
        desired_side: Literal["long", "short"] = "long" if desired > 0 else "short"

        # Safety pre-check: avoid stacking same-side positions and close opposite side first.
        pos = client.get_positions(inst_type=cfg.inst_type, inst_id=cfg.inst_id)
        pdata = pos.get("data") or []
        same_side = 0.0
        opp_side = 0.0
        opposite_close_orders: list[dict[str, Any]] = []
        for p in pdata:
            if p.get("instId") != cfg.inst_id:
                continue
            p_side, p_size = _position_side_and_size(p)
            if p_side is None or p_size <= 0:
                continue
            if p_side == desired_side:
                same_side += p_size
            else:
                opp_side += p_size
                opposite_close_orders.append(_close_order_args(cfg, p_side, p_size))

        if same_side > 0 and opp_side == 0:
            action = "HOLD"
            order_resp = {
                "note": "already in desired-side position, skip duplicate open",
                "desired_side": desired_side,
                "same_side_size": same_side,
            }
        else:
            close_results: list[dict[str, Any]] = []
            if opposite_close_orders:
                if not cfg.enable_trading:
                    close_results = [{"dry_run": True, **o} for o in opposite_close_orders]
                else:
                    for o in opposite_close_orders:
                        close_results.append(
                            client.place_order(
                                inst_id=o["instId"],
                                td_mode=o["tdMode"],
                                side=o["side"],
                                ord_type=o["ordType"],
                                sz=o["sz"],
                                pos_side=o["posSide"],
                                reduce_only=True,
                                lever=None,
                            )
                        )

            if not cfg.enable_trading:
                open_resp: dict[str, Any] = {
                    "dry_run": True,
                    "instId": cfg.inst_id,
                    "tdMode": cfg.td_mode,
                    "side": side,
                    "ordType": "market",
                    "sz": sz,
                    "posSide": pos_side,
                    "lever": lever,
                }
            else:
                open_resp = client.place_order(
                    inst_id=cfg.inst_id,
                    td_mode=cfg.td_mode,
                    side=side,
                    ord_type="market",
                    sz=sz,
                    pos_side=pos_side,
                    reduce_only=False,
                    lever=lever,
                )

            order_resp = {
                "close_opposite_before_open": close_results,
                "open_order": open_resp,
            }

    return {
        "instId": cfg.inst_id,
        "simulated": cfg.simulated,
        "enable_trading": cfg.enable_trading,
        "symbol": symbol,
        "interval": interval,
        "price": price,
        "decision": decision,
        "action": action,
        "leverage": lever,
        "size": sz,
        "set_leverage_response": lev_resp,
        "order_response": order_resp,
        "effective_notional_usdt": float(effective_notional_usdt),
        "risk_controls": {
            "black_swan_score": float(black_swan_score),
            "black_swan_threshold": float(cfg.black_swan_threshold),
            "black_swan_reserve_usdt": float(cfg.black_swan_reserve_usdt),
            "black_swan_active": bool(black_swan_active),
            "hold_due_to_black_swan": bool(hold_due_to_black_swan),
            "panic_news_score": float(panic_news_score),
            "war_news_score": float(war_news_score),
            "macro_event_risk_score": float(macro_event_risk_score),
            "market_panic_score": float(market_panic_score),
            "panic_regime": bool(panic_regime),
            "note": risk_control_note,
            "mtf_gate": {
                "direction": int(mtf_gate.get("direction", 0) or 0),
                "tier": int(mtf_gate.get("tier", 0) or 0),
                "pair": mtf_gate.get("pair"),
                "leverage_cap": int(mtf_gate.get("leverage_cap", 1) or 1),
                "note": mtf_gate.get("note", ""),
            },
        },
    }
