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


def _infer_bar_seconds(ts: pd.Series) -> int:
    if ts is None or ts.empty:
        return 3600
    x = pd.to_datetime(ts, utc=True, errors="coerce").dropna()
    if len(x) < 3:
        return 3600
    diffs = x.diff().dt.total_seconds().dropna()
    if diffs.empty:
        return 3600
    try:
        sec = int(diffs.mode().iloc[0])
    except Exception:
        sec = 3600
    return max(60, sec)


def _snr_window_bars_from_interval(bar_seconds: int) -> list[int]:
    # Multi-timeframe targets used to create SNR-like training features.
    target_secs = [15 * 60, 30 * 60, 60 * 60, 4 * 60 * 60, 24 * 60 * 60]
    out: list[int] = []
    for target in target_secs:
        bars = int(round(target / max(1, bar_seconds)))
        bars = max(3, bars)
        if bars not in out:
            out.append(bars)
    return sorted(out)


def _add_snr_training_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    bar_seconds = _infer_bar_seconds(x.get("timestamp"))
    windows = _snr_window_bars_from_interval(bar_seconds)
    eps = 1e-9

    atr = pd.to_numeric(x.get("atr_14"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    atr_fallback = (pd.to_numeric(x.get("close"), errors="coerce").abs() * 0.002).fillna(1.0)
    atr = atr.fillna(atr_fallback)
    atr = pd.Series(np.where(atr <= 0, atr_fallback, atr), index=x.index, dtype="float64").clip(lower=eps)

    near_s_cols: list[str] = []
    near_r_cols: list[str] = []
    break_s_cols: list[str] = []
    break_r_cols: list[str] = []
    dist_s_cols: list[str] = []
    dist_r_cols: list[str] = []

    for bars in windows:
        min_periods = max(3, int(bars * 0.5))
        s_col = f"snr_support_q10_{bars}"
        r_col = f"snr_resistance_q90_{bars}"
        x[s_col] = x["low"].rolling(bars, min_periods=min_periods).quantile(0.10)
        x[r_col] = x["high"].rolling(bars, min_periods=min_periods).quantile(0.90)

        near_s_col = f"snr_near_support_{bars}"
        near_r_col = f"snr_near_resistance_{bars}"
        break_s_col = f"snr_break_support_{bars}"
        break_r_col = f"snr_break_resistance_{bars}"
        dist_s_col = f"snr_dist_support_atr_{bars}"
        dist_r_col = f"snr_dist_resistance_atr_{bars}"

        x[dist_s_col] = (x["close"] - x[s_col]) / atr
        x[dist_r_col] = (x[r_col] - x["close"]) / atr

        tol = atr * 0.60
        x[near_s_col] = ((x["close"] - x[s_col]).abs() <= tol).astype(int)
        x[near_r_col] = ((x["close"] - x[r_col]).abs() <= tol).astype(int)
        x[break_s_col] = (x["close"] < (x[s_col] - atr * 0.25)).astype(int)
        x[break_r_col] = (x["close"] > (x[r_col] + atr * 0.25)).astype(int)

        near_s_cols.append(near_s_col)
        near_r_cols.append(near_r_col)
        break_s_cols.append(break_s_col)
        break_r_cols.append(break_r_col)
        dist_s_cols.append(dist_s_col)
        dist_r_cols.append(dist_r_col)

    x["snr_overlap_support_count"] = x[near_s_cols].sum(axis=1)
    x["snr_overlap_resistance_count"] = x[near_r_cols].sum(axis=1)
    x["snr_break_support_count"] = x[break_s_cols].sum(axis=1)
    x["snr_break_resistance_count"] = x[break_r_cols].sum(axis=1)
    x["snr_break_pressure"] = x["snr_break_resistance_count"] - x["snr_break_support_count"]

    x["snr_nearest_support_dist_atr"] = x[dist_s_cols].abs().min(axis=1)
    x["snr_nearest_resistance_dist_atr"] = x[dist_r_cols].abs().min(axis=1)
    x["snr_overlap_imbalance"] = x["snr_overlap_support_count"] - x["snr_overlap_resistance_count"]

    return x


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

    x = _add_snr_training_features(x)
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
