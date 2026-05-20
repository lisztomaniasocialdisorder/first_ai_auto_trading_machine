from __future__ import annotations

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV-like klines to a higher timeframe.

    Expects columns: timestamp (UTC), open/high/low/close, volume,
    quote_asset_volume, number_of_trades, taker_buy_base, taker_buy_quote.
    """
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x = x.sort_values("timestamp").set_index("timestamp")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_asset_volume": "sum",
        "number_of_trades": "sum",
        "taker_buy_base": "sum",
        "taker_buy_quote": "sum",
    }
    # Keep any missing numeric columns gracefully.
    agg = {k: v for k, v in agg.items() if k in x.columns}

    out = x.resample(rule, label="left", closed="left").agg(agg).dropna()
    out = out.reset_index()
    return out

