from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone

import pandas as pd


@dataclass
class DriftAlert:
    metric: str
    current: float
    baseline: float
    threshold: float
    triggered: bool
    severity: str  # ok | warning | critical
    message: str


def compute_drift_alerts(
    signals_df: pd.DataFrame,
    window: int = 168,
    confidence_floor: float = 0.45,
) -> list[DriftAlert]:
    """
    Compare the recent window against the full-history baseline and surface
    simple drift signals that are cheap to compute on every dashboard refresh.
    """
    alerts: list[DriftAlert] = []

    if signals_df.empty or len(signals_df) < window:
        return [
            DriftAlert(
                metric="data",
                current=float(len(signals_df)),
                baseline=float(window),
                threshold=float(window),
                triggered=True,
                severity="warning",
                message=f"資料不足：目前只有 {len(signals_df)} 根，低於監控視窗 {window} 根",
            )
        ]

    recent = signals_df.iloc[-window:].copy()
    full = signals_df.copy()

    if "timestamp" in signals_df.columns:
        try:
            latest_ts = pd.to_datetime(signals_df["timestamp"].iloc[-1], utc=True, errors="coerce")
            if not pd.isna(latest_ts):
                now = pd.Timestamp.now(tz=timezone.utc)
                age_seconds = max(0.0, (now - latest_ts).total_seconds())
                if len(signals_df) >= 2:
                    diffs = pd.to_datetime(signals_df["timestamp"], utc=True, errors="coerce").diff().dt.total_seconds().dropna()
                    interval_seconds = float(diffs.median()) if not diffs.empty else 0.0
                else:
                    interval_seconds = 0.0
                stale_threshold = max(3.0 * interval_seconds, 2.0 * 3600.0)
                triggered = age_seconds > stale_threshold
                severity = "critical" if age_seconds > stale_threshold * 2 else ("warning" if triggered else "ok")
                alerts.append(
                    DriftAlert(
                        metric="data_freshness",
                        current=round(age_seconds / 3600.0, 4),
                        baseline=round(stale_threshold / 3600.0, 4),
                        threshold=round(stale_threshold / 3600.0, 4),
                        triggered=triggered,
                        severity=severity,
                        message=f"資料最新時間距今 {age_seconds/3600.0:.2f} 小時，超過容忍門檻 {stale_threshold/3600.0:.2f} 小時",
                    )
                )
        except Exception:
            pass

    if "confidence_index" in signals_df.columns:
        recent_conf = float(pd.to_numeric(recent["confidence_index"], errors="coerce").fillna(0).mean())
        full_conf = float(pd.to_numeric(full["confidence_index"], errors="coerce").fillna(0).mean())
        triggered = recent_conf < confidence_floor
        severity = "critical" if recent_conf < confidence_floor * 0.8 else ("warning" if triggered else "ok")
        alerts.append(
            DriftAlert(
                metric="confidence",
                current=round(recent_conf, 4),
                baseline=round(full_conf, 4),
                threshold=confidence_floor,
                triggered=triggered,
                severity=severity,
                message=f"近期平均信心 {recent_conf:.3f}，低於門檻 {confidence_floor:.3f}",
            )
        )

    if "trade_allowed" in signals_df.columns:
        recent_allowed = float(pd.to_numeric(recent["trade_allowed"], errors="coerce").fillna(0).mean())
        full_allowed = float(pd.to_numeric(full["trade_allowed"], errors="coerce").fillna(0).mean())
        drift = abs(recent_allowed - full_allowed)
        triggered = recent_allowed < 0.25 or drift > 0.25
        severity = "critical" if recent_allowed < 0.15 else ("warning" if triggered else "ok")
        alerts.append(
            DriftAlert(
                metric="trade_allowed_rate",
                current=round(recent_allowed, 4),
                baseline=round(full_allowed, 4),
                threshold=0.25,
                triggered=triggered,
                severity=severity,
                message=f"可交易率 {recent_allowed:.2%}，全期平均 {full_allowed:.2%}，偏移 {drift:.2%}",
            )
        )

    if "signal" in signals_df.columns:
        recent_signal = pd.to_numeric(recent["signal"], errors="coerce").fillna(0)
        full_signal = pd.to_numeric(full["signal"], errors="coerce").fillna(0)
        recent_bull = float((recent_signal == 1).mean())
        full_bull = float((full_signal == 1).mean())
        drift = abs(recent_bull - full_bull)
        triggered = drift > 0.20
        severity = "critical" if drift > 0.35 else ("warning" if triggered else "ok")
        alerts.append(
            DriftAlert(
                metric="signal_distribution",
                current=round(recent_bull, 4),
                baseline=round(full_bull, 4),
                threshold=0.20,
                triggered=triggered,
                severity=severity,
                message=f"近期做多訊號占比 {recent_bull:.2%}，相較基準 {full_bull:.2%} 偏移 {drift:.2%}",
            )
        )

    if "atr_factor" in signals_df.columns:
        recent_atr = float(pd.to_numeric(recent["atr_factor"], errors="coerce").fillna(1).mean())
        full_atr = float(pd.to_numeric(full["atr_factor"], errors="coerce").fillna(1).mean())
        triggered = recent_atr < 0.5
        severity = "critical" if recent_atr < 0.35 else ("warning" if triggered else "ok")
        alerts.append(
            DriftAlert(
                metric="atr_factor",
                current=round(recent_atr, 4),
                baseline=round(full_atr, 4),
                threshold=0.5,
                triggered=triggered,
                severity=severity,
                message=f"ATR 倉位因子 {recent_atr:.3f}，代表波動狀態明顯改變",
            )
        )

    return alerts


def get_system_status(alerts: list[DriftAlert]) -> tuple[str, str]:
    if not alerts:
        return "🟢", "模型狀態穩定"

    criticals = [a for a in alerts if a.severity == "critical" and a.triggered]
    warnings = [a for a in alerts if a.severity == "warning" and a.triggered]

    if criticals:
        metrics = ", ".join(a.metric for a in criticals)
        return "🔴", f"嚴重警報：{metrics}"
    if warnings:
        metrics = ", ".join(a.metric for a in warnings)
        return "🟠", f"注意風險：{metrics}"
    return "🟢", "模型狀態穩定"
