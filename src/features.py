from __future__ import annotations

import numpy as np
import pandas as pd

from .regime import add_regime_features


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values("timestamp").reset_index(drop=True)

    x["ret_1h"] = x["close"].pct_change()
    x["ret_24h"] = x["close"].pct_change(24)

    x["ma_20"] = x["close"].rolling(20).mean()
    x["ma_50"] = x["close"].rolling(50).mean()
    x["ma_100"] = x["close"].rolling(100).mean()
    x["ma_200"] = x["close"].rolling(200).mean()

    x["ema_12"] = _ema(x["close"], 12)
    x["ema_26"] = _ema(x["close"], 26)
    x["macd"] = x["ema_12"] - x["ema_26"]
    x["macd_signal"] = _ema(x["macd"], 9)
    x["macd_hist"] = x["macd"] - x["macd_signal"]

    x["rsi_14"] = _rsi(x["close"], period=14)

    bb_mid = x["close"].rolling(20).mean()
    bb_std = x["close"].rolling(20).std()
    x["bb_mid"] = bb_mid
    x["bb_upper"] = bb_mid + 2 * bb_std
    x["bb_lower"] = bb_mid - 2 * bb_std
    x["bb_width"] = (x["bb_upper"] - x["bb_lower"]) / x["bb_mid"]

    x["atr_14"] = _atr(x, period=14)
    x["atr_pct"] = x["atr_14"] / x["close"]

    direction = np.sign(x["close"].diff().fillna(0))
    x["obv"] = (direction * x["volume"]).cumsum()
    x["vol_ma_20"] = x["volume"].rolling(20).mean()
    x["volume_zscore"] = (x["volume"] - x["vol_ma_20"]) / x["volume"].rolling(20).std()

    x["taker_buy_ratio"] = x["taker_buy_base"] / x["volume"].replace(0, np.nan)
    x["aggr_buy_pressure"] = (x["taker_buy_quote"] - (x["quote_asset_volume"] - x["taker_buy_quote"])) / x[
        "quote_asset_volume"
    ].replace(0, np.nan)
    x["spot_liquidity_proxy"] = x["quote_asset_volume"]
    x["orderbook_depth_imbalance_proxy"] = x["aggr_buy_pressure"]
    x["large_order_flow_proxy"] = x["volume_zscore"] * x["taker_buy_ratio"]

    x["rolling_high_24"] = x["high"].rolling(24).max()
    x["rolling_low_24"] = x["low"].rolling(24).min()
    x["rolling_high_168"] = x["high"].rolling(168).max()
    x["rolling_low_168"] = x["low"].rolling(168).min()

    x["support_48"] = x["low"].rolling(48).quantile(0.1)
    x["resistance_48"] = x["high"].rolling(48).quantile(0.9)
    x["dist_to_support"] = (x["close"] - x["support_48"]) / x["close"]
    x["dist_to_resistance"] = (x["resistance_48"] - x["close"]) / x["close"]

    x["realized_vol_24"] = x["ret_1h"].rolling(24).std() * np.sqrt(24)
    x["realized_vol_168"] = x["ret_1h"].rolling(168).std() * np.sqrt(168)

    x["equity_curve_proxy"] = (1 + x["ret_1h"].fillna(0)).cumprod()
    x["rolling_peak"] = x["equity_curve_proxy"].cummax()
    x["drawdown"] = x["equity_curve_proxy"] / x["rolling_peak"] - 1

    x["trend_strength"] = (x["ma_20"] - x["ma_100"]) / x["close"]
    x["is_uptrend"] = (x["ma_20"] > x["ma_50"]).astype(int)
    x["is_downtrend"] = (x["ma_20"] < x["ma_50"]).astype(int)

    x = add_regime_features(x)
    return x


def build_labels(
    df: pd.DataFrame,
    horizon_bars: int,
    long_th: float,
    short_th: float,
) -> pd.DataFrame:
    x = df.copy()
    horizon_bars = int(max(1, horizon_bars))
    x["future_ret"] = x["close"].shift(-horizon_bars) / x["close"] - 1

    conditions = [x["future_ret"] >= long_th, x["future_ret"] <= short_th]
    choices = [1, -1]
    x["label"] = np.select(conditions, choices, default=0)

    x["target_leverage"] = (
        (x["future_ret"].abs() / x["atr_pct"].replace(0, np.nan)).clip(lower=0.5, upper=25)
    )
    x["target_leverage"] = x["target_leverage"].fillna(1.0)
    return x
