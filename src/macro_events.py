from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

NY_TZ = ZoneInfo("America/New_York")
UTC = timezone.utc


@dataclass(frozen=True)
class MacroEvent:
    timestamp: pd.Timestamp
    event_name: str
    event_type: str
    risk_weight: float
    source: str = "estimated_calendar"


def _month_range(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    y, m = int(start_ts.year), int(start_ts.month)
    end_y, end_m = int(end_ts.year), int(end_ts.month)
    while (y < end_y) or (y == end_y and m <= end_m):
        out.append((y, m))
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def _nth_weekday_of_month(year: int, month: int, weekday: int, nth: int) -> date:
    # weekday: Monday=0 ... Sunday=6
    d = date(year, month, 1)
    shift = (weekday - d.weekday()) % 7
    day = 1 + shift + (nth - 1) * 7
    return date(year, month, day)


def _next_business_day(d: date) -> date:
    out = d + timedelta(days=1)
    while out.weekday() >= 5:
        out += timedelta(days=1)
    return out


def _ny_time_to_utc_ts(d: date, hour: int, minute: int = 0) -> pd.Timestamp:
    local_dt = datetime(d.year, d.month, d.day, hour, minute, tzinfo=NY_TZ)
    return pd.Timestamp(local_dt.astimezone(UTC))


def generate_estimated_macro_events(
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> pd.DataFrame:
    start = pd.to_datetime(start_utc, utc=True)
    end = pd.to_datetime(end_utc, utc=True)
    if pd.isna(start) or pd.isna(end) or start >= end:
        return pd.DataFrame(columns=["timestamp", "event_name", "event_type", "risk_weight", "source"])

    # Expand search window a bit to capture boundary events.
    start_local = (start - pd.Timedelta(days=40)).tz_convert(NY_TZ)
    end_local = (end + pd.Timedelta(days=40)).tz_convert(NY_TZ)
    ym_list = _month_range(pd.Timestamp(start_local), pd.Timestamp(end_local))

    events: list[MacroEvent] = []

    # CPI: assume 2nd Tuesday 08:30 ET
    # PPI: assume next business day after CPI 08:30 ET
    for y, m in ym_list:
        cpi_d = _nth_weekday_of_month(y, m, weekday=1, nth=2)  # Tuesday
        ppi_d = _next_business_day(cpi_d)
        cpi_ts = _ny_time_to_utc_ts(cpi_d, 8, 30)
        ppi_ts = _ny_time_to_utc_ts(ppi_d, 8, 30)
        events.append(MacroEvent(timestamp=cpi_ts, event_name="US CPI 公布（預估）", event_type="CPI", risk_weight=0.8))
        events.append(MacroEvent(timestamp=ppi_ts, event_name="US PPI 公布（預估）", event_type="PPI", risk_weight=0.7))

    # FOMC conference: assume 3rd Wednesday in common FOMC months 14:30 ET
    fomc_months = {1, 3, 5, 6, 7, 9, 11, 12}
    for y, m in ym_list:
        if m not in fomc_months:
            continue
        fomc_d = _nth_weekday_of_month(y, m, weekday=2, nth=3)  # Wednesday
        fomc_ts = _ny_time_to_utc_ts(fomc_d, 14, 30)
        events.append(MacroEvent(timestamp=fomc_ts, event_name="FOMC 記者會（預估）", event_type="FOMC", risk_weight=1.0))

    if not events:
        return pd.DataFrame(columns=["timestamp", "event_name", "event_type", "risk_weight", "source"])

    out = pd.DataFrame(
        {
            "timestamp": [e.timestamp for e in events],
            "event_name": [e.event_name for e in events],
            "event_type": [e.event_type for e in events],
            "risk_weight": [float(e.risk_weight) for e in events],
            "source": [e.source for e in events],
        }
    )
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out[(out["timestamp"] >= start) & (out["timestamp"] <= end)].sort_values("timestamp").reset_index(drop=True)
    return out

