from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SnrLevel:
    price: float
    kinds: set[str]  # {"S", "R"} support/resistance
    timeframes: set[str]  # {"5m","15m","1h","1d"}


def _atr_proxy(df: pd.DataFrame, period: int = 14) -> float:
    """Lightweight ATR proxy used only for level merging tolerance."""
    x = df.copy()
    high_low = x["high"] - x["low"]
    high_close = (x["high"] - x["close"].shift(1)).abs()
    low_close = (x["low"] - x["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    v = float(atr.dropna().tail(100).median()) if not atr.dropna().empty else float(tr.dropna().median())
    return max(v, 1e-6)


def _pivot_points(series: pd.Series, window: int, mode: str) -> pd.Series:
    """
    mode: "high" or "low"
    Returns series with pivot values at pivot index, else NaN.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = series
    left = s.shift(1).rolling(window).max() if mode == "high" else s.shift(1).rolling(window).min()
    right = s.shift(-window).rolling(window).max() if mode == "high" else s.shift(-window).rolling(window).min()

    if mode == "high":
        piv = (s >= left) & (s >= right)
    else:
        piv = (s <= left) & (s <= right)
    out = s.where(piv)
    return out


def compute_snr_levels(
    df: pd.DataFrame,
    timeframe: str,
    lookback_bars: int = 800,
    pivot_window: int = 5,
    max_levels: int = 8,
    merge_tolerance_atr: float = 0.6,
) -> list[SnrLevel]:
    """
    Pivot-based S/R detection + clustering merge.
    Output levels are approximate, intended for UI overlays not execution.
    """
    x = df.copy().sort_values("timestamp").tail(int(lookback_bars)).reset_index(drop=True)
    if x.empty:
        return []

    atr = _atr_proxy(x, period=14)
    tol = atr * float(merge_tolerance_atr)

    piv_hi = _pivot_points(x["high"], window=pivot_window, mode="high").dropna()
    piv_lo = _pivot_points(x["low"], window=pivot_window, mode="low").dropna()

    candidates: list[tuple[float, str]] = []
    candidates += [(float(v), "R") for v in piv_hi.to_list()]
    candidates += [(float(v), "S") for v in piv_lo.to_list()]
    if not candidates:
        return []

    # Frequency by proximity (cluster count). This tends to surface repeated levels.
    prices = np.array([p for p, _ in candidates], dtype=float)
    kinds = [k for _, k in candidates]

    clusters: list[dict] = []
    order = np.argsort(prices)
    for idx in order:
        p = float(prices[idx])
        k = kinds[idx]
        placed = False
        for c in clusters:
            if abs(p - c["price"]) <= tol:
                # merge into cluster center (mean)
                c["prices"].append(p)
                c["price"] = float(np.mean(c["prices"]))
                c["kinds"].add(k)
                placed = True
                break
        if not placed:
            clusters.append({"price": p, "prices": [p], "kinds": {k}})

    # Score by number of touches and proximity to current price.
    last = float(x["close"].iloc[-1])
    for c in clusters:
        touches = len(c["prices"])
        dist = abs(c["price"] - last) / max(last, 1e-9)
        c["score"] = touches * (1.0 / (dist + 0.01))

    clusters.sort(key=lambda c: c["score"], reverse=True)
    clusters = clusters[: int(max_levels)]

    out: list[SnrLevel] = []
    for c in clusters:
        out.append(SnrLevel(price=float(c["price"]), kinds=set(c["kinds"]), timeframes={timeframe}))
    return out


def merge_multitimeframe_levels(
    levels: Iterable[SnrLevel],
    tolerance_abs: float,
) -> list[SnrLevel]:
    """Merge levels across timeframes if they are within abs tolerance."""
    merged: list[SnrLevel] = []
    for lv in sorted(levels, key=lambda x: x.price):
        placed = False
        for i, m in enumerate(merged):
            if abs(lv.price - m.price) <= tolerance_abs:
                new_price = float(np.mean([m.price, lv.price]))
                merged[i] = SnrLevel(
                    price=new_price,
                    kinds=set(m.kinds) | set(lv.kinds),
                    timeframes=set(m.timeframes) | set(lv.timeframes),
                )
                placed = True
                break
        if not placed:
            merged.append(lv)
    return merged

