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
    sig = int(decision.get("signal", 0))
    suggested_lev = float(decision.get("suggested_leverage", 1.0))
    price = float(decision.get("price", 0.0)) or _load_latest_price(outputs_dir, symbol, interval)

    # Decide desired action
    mode = action_override or "AUTO"
    if mode == "LONG":
        desired = 1
    elif mode == "SHORT":
        desired = -1
    elif mode == "CLOSE":
        desired = 0
    else:
        desired = 1 if sig > 0 else (-1 if sig < 0 else 0)

    lever = int(leverage_override or min(cfg.max_leverage, max(1, int(round(suggested_lev)))))

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

    sz = _calc_swap_contract_size(client, cfg, price) if cfg.inst_type in ("SWAP", "FUTURES") else str(cfg.notional_usdt)

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
                p_pos = float(p.get("pos") or 0)
                if p_pos == 0:
                    continue
                p_side = str(p.get("posSide") or "").lower()  # long/short or empty (net)
                if p_side == "long":
                    side = "sell"
                    ps = "long" if cfg.pos_mode == "long_short" else None
                    sz = str(abs(p_pos))
                elif p_side == "short":
                    side = "buy"
                    ps = "short" if cfg.pos_mode == "long_short" else None
                    sz = str(abs(p_pos))
                else:
                    # net mode: if we cannot infer, skip.
                    continue

                if not cfg.enable_trading:
                    close_orders.append(
                        {
                            "dry_run": True,
                            "instId": cfg.inst_id,
                            "tdMode": cfg.td_mode,
                            "side": side,
                            "ordType": "market",
                            "sz": sz,
                            "posSide": ps,
                            "reduceOnly": True,
                        }
                    )
                else:
                    close_orders.append(
                        client.place_order(
                            inst_id=cfg.inst_id,
                            td_mode=cfg.td_mode,
                            side=side,
                            ord_type="market",
                            sz=sz,
                            pos_side=ps,
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
    else:
        action = "OPEN_LONG" if desired > 0 else "OPEN_SHORT"
        side: Literal["buy", "sell"] = "buy" if desired > 0 else "sell"
        if not cfg.enable_trading:
            order_resp = {"dry_run": True, "instId": cfg.inst_id, "tdMode": cfg.td_mode, "side": side, "ordType": "market", "sz": sz, "posSide": pos_side, "lever": lever}
        else:
            order_resp = client.place_order(
                inst_id=cfg.inst_id,
                td_mode=cfg.td_mode,
                side=side,
                ord_type="market",
                sz=sz,
                pos_side=pos_side,
                reduce_only=False,
                lever=lever,
            )

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
    }
