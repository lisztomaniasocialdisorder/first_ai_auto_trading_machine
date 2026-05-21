from __future__ import annotations
import numpy as np
import pandas as pd


def _adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute ADX, +DI, -DI."""
    high = df['high']
    low = df['low']
    close = df['close']

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    result = df.copy()
    result['adx'] = adx.round(2)
    result['plus_di'] = plus_di.round(2)
    result['minus_di'] = minus_di.round(2)
    return result


def classify_regime(df: pd.DataFrame, adx_period: int = 14) -> pd.Series:
    """
    Classify each bar into a market regime:
      'trend'    -> ADX > 25 with clear directional bias
      'volatile' -> atr_pct > 80th percentile historically
      'ranging'  -> everything else (choppy/sideways)

    Returns a pd.Series of regime strings, index aligned to df.
    """
    if 'adx' not in df.columns:
        df = _adx(df, period=adx_period)

    adx = df['adx'].fillna(0)

    if 'atr_pct' in df.columns:
        atr_pct = df['atr_pct']
    elif 'atr_14' in df.columns and 'close' in df.columns:
        atr_pct = (df['atr_14'] / df['close'].replace(0, np.nan)).fillna(0)
    else:
        atr_pct = pd.Series(0.01, index=df.index)

    atr_80th = atr_pct.rolling(500, min_periods=50).quantile(0.80).fillna(atr_pct.median())

    regime = pd.Series('ranging', index=df.index, dtype=str)
    regime[atr_pct > atr_80th] = 'volatile'
    regime[adx > 25] = 'trend'

    return regime


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ADX + regime one-hot columns to df in-place."""
    result = _adx(df)
    regime = classify_regime(result)
    result['regime'] = regime
    result['regime_trend'] = (regime == 'trend').astype(int)
    result['regime_volatile'] = (regime == 'volatile').astype(int)
    result['regime_ranging'] = (regime == 'ranging').astype(int)
    return result
