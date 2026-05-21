from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .data_sources import interval_to_seconds


def extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build trade-level records from per-bar position series.
    Expects columns: timestamp, close, position (signal shifted), position_lev, strategy_ret, trading_cost.
    """
    x = df.copy().sort_values("timestamp").reset_index(drop=True)
    pos = x["position"].fillna(0).to_numpy()

    entries = []
    in_trade = False
    entry_i = 0
    entry_side = 0
    for i in range(1, len(x)):
        prev = int(np.sign(pos[i - 1]))
        curr = int(np.sign(pos[i]))
        if not in_trade and prev == 0 and curr != 0:
            in_trade = True
            entry_i = i
            entry_side = curr
        elif in_trade:
            # Exit when position goes flat or flips.
            if curr == 0 or curr != entry_side:
                exit_i = i
                entries.append((entry_i, exit_i, entry_side))
                in_trade = False
                if curr != 0:
                    # immediate flip => new entry at same bar
                    in_trade = True
                    entry_i = i
                    entry_side = curr

    if in_trade:
        entries.append((entry_i, len(x) - 1, entry_side))

    rows = []
    for en, ex, side in entries:
        entry_ts = x.loc[en, "timestamp"]
        exit_ts = x.loc[ex, "timestamp"]
        entry_px = float(x.loc[en, "close"])
        exit_px = float(x.loc[ex, "close"])
        lev = float(abs(x.loc[en, "position_lev"])) if "position_lev" in x.columns else 1.0

        # Sum per-bar strategy returns during the holding window (inclusive).
        seg = x.loc[en:ex, "strategy_ret"].fillna(0.0)
        seg_cost = x.loc[en:ex, "trading_cost"].fillna(0.0) if "trading_cost" in x.columns else 0.0
        trade_ret = float((1.0 + seg).prod() - 1.0)
        trade_cost = float(seg_cost.sum()) if hasattr(seg_cost, "sum") else 0.0

        # Directional price return for reference.
        px_ret = (exit_px / entry_px - 1.0) * (1.0 if side > 0 else -1.0)

        # MFE/MAE using close-only proxy
        closes = x.loc[en:ex, "close"].astype(float)
        rel = (closes / entry_px - 1.0) * (1.0 if side > 0 else -1.0)
        mfe = float(rel.max())
        mae = float(rel.min())

        rows.append(
            {
                "進場時間": str(entry_ts),
                "出場時間": str(exit_ts),
                "方向": "多" if side > 0 else "空",
                "槓桿": round(lev, 2),
                "進場價": entry_px,
                "出場價": exit_px,
                "價格報酬(方向)": px_ret,
                "策略報酬(含費用)": trade_ret,
                "費用估計": trade_cost,
                "MFE(最大有利)": mfe,
                "MAE(最大不利)": mae,
                "持倉K數": int(ex - en + 1),
            }
        )

    return pd.DataFrame(rows)


def run_backtest(signal_df: pd.DataFrame, settings: Settings, interval: str | None = None) -> tuple[pd.DataFrame, dict]:
    df = signal_df.copy().sort_values("timestamp").reset_index(drop=True)

    _interval = interval or getattr(settings, 'interval', '1h') or '1h'
    try:
        _bars_per_year = (365 * 24 * 3600) / max(1, interval_to_seconds(_interval))
    except Exception:
        _bars_per_year = 24 * 365
    annual_factor = np.sqrt(_bars_per_year)

    df["bar_ret"] = df["close"].pct_change().fillna(0)
    df["position"] = df["signal"].shift(1).fillna(0)

    lev = df["suggested_leverage"].shift(1).fillna(1.0).clip(lower=1, upper=settings.max_leverage)
    df["position_lev"] = df["position"] * lev

    turnover = (df["position"] - df["position"].shift(1).fillna(0)).abs()
    trading_cost = turnover * (settings.fee_bps + settings.slippage_bps) / 10_000
    df["turnover"] = turnover
    df["trading_cost"] = trading_cost

    try:
        _funding_rate_8h = float(getattr(settings, 'funding_rate_8h_bps', 2.5) or 2.5) / 10_000
        _interval_sec = max(1, interval_to_seconds(_interval))
        _bars_per_8h = max(1.0, (8 * 3600) / _interval_sec)
        funding_cost = df["position"].abs() * lev * (_funding_rate_8h / _bars_per_8h)
    except Exception:
        funding_cost = pd.Series(0.0, index=df.index)
    df["funding_cost"] = funding_cost

    df["strategy_ret"] = (df["position_lev"] * df["bar_ret"]) - trading_cost - funding_cost

    # Drawdown kill-switch
    equity = (1 + df["strategy_ret"]).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1

    kill = dd < -settings.drawdown_stop
    if kill.any():
        first_kill_idx = int(np.argmax(kill.to_numpy()))
        df.loc[first_kill_idx:, "position_lev"] = 0
        df.loc[first_kill_idx:, "strategy_ret"] = -trading_cost.loc[first_kill_idx:]
        equity = (1 + df["strategy_ret"]).cumprod()
        peak = equity.cummax()
        dd = equity / peak - 1

    df["equity"] = equity
    df["drawdown_curve"] = dd

    trade_mask = turnover > 0
    realized = df.loc[trade_mask, "strategy_ret"]
    wins = realized[realized > 0]
    losses = realized[realized < 0]

    gross_profit = wins.sum() if not wins.empty else 0.0
    gross_loss = -losses.sum() if not losses.empty else 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
    win_rate = (realized > 0).mean() if len(realized) else 0.0
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = -losses.mean() if len(losses) else 0.0
    pnl_ratio = avg_win / avg_loss if avg_loss > 0 else np.nan

    total_return = equity.iloc[-1] - 1
    # annual_factor computed dynamically above
    sharpe = df["strategy_ret"].mean() / (df["strategy_ret"].std() + 1e-12) * annual_factor

    downside = df["strategy_ret"].copy()
    downside[downside > 0] = 0
    sortino = df["strategy_ret"].mean() / (downside.std() + 1e-12) * annual_factor

    max_dd = float(dd.min())
    calmar = (float(total_return) / abs(max_dd)) if max_dd < 0 else np.nan

    # Historical daily-ish VaR/ES on per-bar returns (approx, still useful).
    r = df["strategy_ret"].dropna().to_numpy()
    var_95 = float(np.quantile(r, 0.05)) if len(r) else 0.0
    es_95 = float(r[r <= var_95].mean()) if len(r[r <= var_95]) else var_95

    report = {
        "rows": int(len(df)),
        "trades": int(len(realized)),
        "total_return": float(total_return),
        "max_drawdown": max_dd,
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor) if not np.isnan(profit_factor) else None,
        "pnl_ratio": float(pnl_ratio) if not np.isnan(pnl_ratio) else None,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar) if not np.isnan(calmar) else None,
        "var_95": var_95,
        "es_95": es_95,
        "avg_leverage": float(lev.mean()),
        "max_leverage_used": float(lev.max()),
        "funding_rate_8h_bps": float(getattr(settings, 'funding_rate_8h_bps', 2.5) or 2.5),
    }
    return df, report
