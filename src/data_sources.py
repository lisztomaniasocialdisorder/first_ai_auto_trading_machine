from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from .config import Settings
from .cache import JsonCache
from .macro_events import generate_estimated_macro_events


INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def interval_to_seconds(interval: str) -> int:
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    return int(INTERVAL_TO_MS[interval] // 1000)


@dataclass
class BinanceClient:
    settings: Settings

    @property
    def kline_url(self) -> str:
        if self.settings.market_type == "futures":
            return "https://fapi.binance.com/fapi/v1/klines"
        return f"{self.settings.base_url}/api/v3/klines"

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        if interval not in INTERVAL_TO_MS:
            raise ValueError(f"Unsupported interval: {interval}")

        limit = limit or self.settings.kline_limit
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)

        resp = requests.get(self.kline_url, params=params, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            return pd.DataFrame()

        cols = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        df = pd.DataFrame(raw, columns=cols)
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
        ]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.drop(columns=["ignore"]).sort_values("timestamp").reset_index(drop=True)
        return df

    def fetch_all_history(self, symbol: str, interval: str, start_ms: int) -> pd.DataFrame:
        step_ms = INTERVAL_TO_MS[interval] * self.settings.kline_limit
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

        cursor = start_ms
        chunks: list[pd.DataFrame] = []
        while cursor < now_ms:
            batch = self.fetch_klines(symbol, interval, start_ms=cursor, limit=self.settings.kline_limit)
            if batch.empty:
                break
            chunks.append(batch)
            last_open_time = int(batch["open_time"].iloc[-1])
            next_cursor = last_open_time + INTERVAL_TO_MS[interval]
            if next_cursor <= cursor:
                break
            cursor = next_cursor

            if len(batch) < self.settings.kline_limit:
                break
            if cursor - start_ms > 250_000 * step_ms:
                break

        if not chunks:
            return pd.DataFrame()

        out = pd.concat(chunks, ignore_index=True)
        out = out.drop_duplicates(subset=["open_time"]).sort_values("timestamp").reset_index(drop=True)
        return out


def load_or_update_ohlcv(settings: Settings, start_ms: int, force_full_refresh: bool = False) -> pd.DataFrame:
    csv_path = settings.data_dir / f"{settings.symbol}_{settings.interval}_ohlcv.csv"
    pq_path = settings.data_dir / f"{settings.symbol}_{settings.interval}_ohlcv.parquet"
    client = BinanceClient(settings)
    try:
        keep_rows = max(0, int(os.getenv("KLINE_KEEP_ROWS", "0")))
    except Exception:
        keep_rows = 0

    if force_full_refresh or (not csv_path.exists() and not pq_path.exists()):
        df = client.fetch_all_history(settings.symbol, settings.interval, start_ms=start_ms)
        if df.empty:
            raise RuntimeError("No OHLCV data fetched in full refresh.")
        if keep_rows > 0:
            df = df.sort_values("timestamp").tail(keep_rows).reset_index(drop=True)
        df.to_parquet(pq_path, index=False)
        df.to_csv(csv_path, index=False)
        return df

    if pq_path.exists():
        old = pd.read_parquet(pq_path)
    else:
        old = pd.read_csv(csv_path)
    old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
    old = old.sort_values("timestamp").reset_index(drop=True)

    interval_ms = INTERVAL_TO_MS[settings.interval]
    overlap_ms = settings.hours_lookback_overlap * 3_600_000
    from_ms = int(old["open_time"].iloc[-1]) - overlap_ms
    from_ms = max(from_ms, int(old["open_time"].iloc[0]))

    new_tail = client.fetch_all_history(settings.symbol, settings.interval, start_ms=from_ms)
    if new_tail.empty:
        if keep_rows > 0 and len(old) > keep_rows:
            old = old.sort_values("timestamp").tail(keep_rows).reset_index(drop=True)
            old.to_parquet(pq_path, index=False)
            old.to_csv(csv_path, index=False)
        return old

    keep = old[old["open_time"] < from_ms]
    merged = pd.concat([keep, new_tail], ignore_index=True)
    merged = merged.drop_duplicates(subset=["open_time"], keep="last")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    if keep_rows > 0:
        merged = merged.tail(keep_rows).reset_index(drop=True)

    merged.to_parquet(pq_path, index=False)
    merged.to_csv(csv_path, index=False)
    return merged


def fetch_fear_greed_daily(limit: int = 2000, cache_path: str | None = None, max_age_seconds: int = 6 * 3600) -> pd.DataFrame:
    url = "https://api.alternative.me/fng/"
    if cache_path:
        cache = JsonCache(Path(cache_path))
        if cache.is_fresh(max_age_seconds):
            cached = cache.read()
            if cached and "data" in cached:
                rows = cached.get("data", [])
            else:
                rows = []
        else:
            resp = requests.get(url, params={"limit": limit, "format": "json"}, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("data", [])
            cache.write(payload)
    else:
        resp = requests.get(url, params={"limit": limit, "format": "json"}, timeout=20)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
    if not rows:
        return pd.DataFrame(columns=["date", "fear_greed_value"])

    fg = pd.DataFrame(rows)
    fg["timestamp"] = pd.to_datetime(pd.to_numeric(fg["timestamp"], errors="coerce"), unit="s", utc=True)
    fg["fear_greed_value"] = pd.to_numeric(fg["value"], errors="coerce")
    fg["date"] = fg["timestamp"].dt.floor("D")
    return fg[["date", "fear_greed_value"]].dropna().drop_duplicates(subset=["date"]).sort_values("date")


def _simple_sentiment_score(text: str) -> float:
    bullish = ["etf", "approval", "adoption", "inflow", "rally", "upgrade", "partnership"]
    bearish = ["ban", "hack", "lawsuit", "outflow", "liquidation", "crash", "fraud"]

    t = text.lower()
    b = sum(1 for k in bullish if k in t)
    s = sum(1 for k in bearish if k in t)
    return float(b - s)


def _topic_scores(text: str) -> dict[str, float]:
    t = text.lower()

    etf_kw = ["etf", "sec", "fund", "inflow", "outflow"]
    reg_kw = ["regulation", "regulator", "law", "lawsuit", "ban", "compliance", "sec"]
    exch_kw = ["binance", "coinbase", "kraken", "exchange", "delist", "listing", "maintenance", "outage"]
    black_swan_kw = ["hack", "bankrupt", "collapse", "exploit", "fraud", "war", "sanction", "liquidation cascade"]
    fed_kw = ["federal reserve", "fed", "fomc", "powell", "rate hike", "rate cut", "美聯儲", "聯準會"]
    trump_kw = ["trump", "donald trump", "川普"]
    panic_kw = ["panic", "bank run", "credit stress", "liquidity crunch", "contagion", "fear spike", "恐慌", "擠兌", "流動性危機"]
    war_kw = ["war", "invasion", "missile", "airstrike", "military conflict", "sanction", "戰爭", "衝突", "制裁"]

    def score(keys: list[str]) -> float:
        return float(sum(1 for k in keys if k in t))

    return {
        "etf_news_score": score(etf_kw),
        "regulatory_news_score": score(reg_kw),
        "exchange_event_score": score(exch_kw),
        "black_swan_risk_score": score(black_swan_kw),
        "fed_news_score": score(fed_kw),
        "trump_news_score": score(trump_kw),
        "panic_news_score": score(panic_kw),
        "war_news_score": score(war_kw),
    }


def fetch_cryptopanic_news(
    token: str,
    currency: str = "BTC",
    pages: int = 3,
    cache_path: str | None = None,
    max_age_seconds: int = 15 * 60,
) -> pd.DataFrame:
    if not token:
        return pd.DataFrame(
            columns=[
                "published_at",
                "news_sentiment",
                "news_shock",
                "etf_news_score",
                "regulatory_news_score",
                "exchange_event_score",
                "black_swan_risk_score",
                "fed_news_score",
                "trump_news_score",
                "panic_news_score",
                "war_news_score",
            ]
        )

    base_url = "https://cryptopanic.com/api/v1/posts/"
    all_rows: list[dict] = []

    if cache_path:
        cache = JsonCache(Path(cache_path))
        if cache.is_fresh(max_age_seconds):
            cached = cache.read()
            if cached and "rows" in cached:
                out = pd.DataFrame(cached["rows"])
                if not out.empty:
                    out["published_at"] = pd.to_datetime(out["published_at"], utc=True)
                    return out.sort_values("published_at").reset_index(drop=True)

    for page in range(1, pages + 1):
        params = {
            "auth_token": token,
            "currencies": currency,
            "kind": "news",
            "page": page,
            "public": "true",
        }
        resp = requests.get(base_url, params=params, timeout=20)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            break

        for row in results:
            title = str(row.get("title") or "")
            body = str(row.get("body") or "")
            text_all = f"{title} {body}".strip()
            published = row.get("published_at")
            score = _simple_sentiment_score(text_all)
            shock = 1 if abs(score) >= 2 else 0
            topics = _topic_scores(text_all)
            all_rows.append(
                {
                    "published_at": published,
                    "news_sentiment": score,
                    "news_shock": shock,
                    "title": title,
                    "body": body,
                    **topics,
                }
            )

    if not all_rows:
        return pd.DataFrame(columns=["published_at", "news_sentiment", "news_shock"])

    out = pd.DataFrame(all_rows)
    out["published_at"] = pd.to_datetime(out["published_at"], utc=True)
    if cache_path:
        JsonCache(Path(cache_path)).write({"rows": out.to_dict(orient="records")})
    return out.sort_values("published_at").reset_index(drop=True)


def merge_event_features(price_df: pd.DataFrame, settings: Settings, fast_mode: bool = False) -> pd.DataFrame:
    df = price_df.copy()
    df["date"] = df["timestamp"].dt.floor("D")

    fg_cache = str(settings.data_dir / "cache_fng.json")
    fg = fetch_fear_greed_daily(limit=3000, cache_path=fg_cache, max_age_seconds=6 * 3600)
    df = df.merge(fg, on="date", how="left")
    df["fear_greed_value"] = df["fear_greed_value"].ffill().bfill()

    # Keep checking macro/news flow regularly for both normal and quick update flows.
    news_cache = str(settings.data_dir / "cache_cryptopanic.json")
    _pages = 4 if fast_mode else 10
    news = fetch_cryptopanic_news(
        settings.cryptopanic_auth_token,
        currency="BTC",
        pages=_pages,
        cache_path=news_cache,
        max_age_seconds=60 * 60,  # hourly refresh
    )
    if not news.empty:
        news["hour"] = news["published_at"].dt.floor("h")
        hourly_news = (
            news.groupby("hour", as_index=False)
            .agg(
                news_sentiment=("news_sentiment", "mean"),
                news_shock=("news_shock", "max"),
                etf_news_score=("etf_news_score", "mean"),
                regulatory_news_score=("regulatory_news_score", "mean"),
                exchange_event_score=("exchange_event_score", "mean"),
                black_swan_risk_score=("black_swan_risk_score", "mean"),
                fed_news_score=("fed_news_score", "sum"),
                trump_news_score=("trump_news_score", "sum"),
                panic_news_score=("panic_news_score", "sum"),
                war_news_score=("war_news_score", "sum"),
            )
            .rename(columns={"hour": "timestamp"})
        )
        # Persist hourly news features for traceability/training audits.
        try:
            tag = f"{settings.symbol}_{settings.interval}"
            hourly_news.to_csv(settings.data_dir / f"news_hourly_{tag}.csv", index=False, encoding="utf-8")
        except Exception:
            pass
        df = df.merge(hourly_news, on="timestamp", how="left")
    else:
        df["news_sentiment"] = np.nan
        df["news_shock"] = np.nan
        df["etf_news_score"] = np.nan
        df["regulatory_news_score"] = np.nan
        df["exchange_event_score"] = np.nan
        df["black_swan_risk_score"] = np.nan
        df["fed_news_score"] = np.nan
        df["trump_news_score"] = np.nan
        df["panic_news_score"] = np.nan
        df["war_news_score"] = np.nan

    # Add estimated fixed-time macro events (CPI/PPI/FOMC) to features.
    _start = pd.to_datetime(df["timestamp"].min(), utc=True)
    _end = pd.to_datetime(df["timestamp"].max(), utc=True) + pd.Timedelta(days=120)
    macro_events = generate_estimated_macro_events(_start, _end)
    if not macro_events.empty:
        macro_events = macro_events.copy()
        macro_events["timestamp"] = pd.to_datetime(macro_events["timestamp"], utc=True).dt.floor("h")
        macro_hourly = (
            macro_events.groupby("timestamp", as_index=False)
            .agg(
                macro_event_count=("event_name", "count"),
                macro_event_risk_score=("risk_weight", "sum"),
            )
        )
        df = df.merge(macro_hourly, on="timestamp", how="left")
        # Time to next macro event in hours.
        ev_ts = pd.to_datetime(macro_events["timestamp"], utc=True).sort_values().unique()
        if len(ev_ts) > 0:
            ev_index = pd.DatetimeIndex(ev_ts)
            ts_arr = pd.to_datetime(df["timestamp"], utc=True)
            next_pos = ev_index.searchsorted(ts_arr, side="left")
            hours_to_next: list[float] = []
            for i, pos in enumerate(next_pos):
                if pos >= len(ev_index):
                    hours_to_next.append(np.nan)
                else:
                    delta_h = (ev_index[pos] - ts_arr.iloc[i]).total_seconds() / 3600.0
                    hours_to_next.append(float(delta_h))
            df["hours_to_next_macro_event"] = hours_to_next
        else:
            df["hours_to_next_macro_event"] = np.nan
    else:
        df["macro_event_count"] = np.nan
        df["macro_event_risk_score"] = np.nan
        df["hours_to_next_macro_event"] = np.nan

    df["news_sentiment"] = df["news_sentiment"].fillna(0.0)
    df["news_shock"] = df["news_shock"].fillna(0)
    df["etf_news_score"] = df["etf_news_score"].fillna(0.0)
    df["regulatory_news_score"] = df["regulatory_news_score"].fillna(0.0)
    df["exchange_event_score"] = df["exchange_event_score"].fillna(0.0)
    df["black_swan_risk_score"] = df["black_swan_risk_score"].fillna(0.0)
    df["fed_news_score"] = df["fed_news_score"].fillna(0.0)
    df["trump_news_score"] = df["trump_news_score"].fillna(0.0)
    df["panic_news_score"] = df["panic_news_score"].fillna(0.0)
    df["war_news_score"] = df["war_news_score"].fillna(0.0)
    df["macro_event_count"] = df["macro_event_count"].fillna(0.0)
    df["macro_event_risk_score"] = df["macro_event_risk_score"].fillna(0.0)
    df["hours_to_next_macro_event"] = pd.to_numeric(df["hours_to_next_macro_event"], errors="coerce")
    df["macro_event_within_24h"] = ((df["hours_to_next_macro_event"] >= 0) & (df["hours_to_next_macro_event"] <= 24)).astype(float)

    # Composite market-panic signal for trading/risk control.
    df["market_panic_score"] = (
        1.2 * df["panic_news_score"]
        + 1.5 * df["war_news_score"]
        + 0.8 * df["black_swan_risk_score"]
        + 0.4 * df["macro_event_risk_score"]
        + np.where(df["fear_greed_value"] <= 25, 1.0, 0.0)
    )
    return df
