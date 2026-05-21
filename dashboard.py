from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from plotly.subplots import make_subplots

from src.backtest import run_backtest
from src.config import Settings
from src.monitor import compute_drift_alerts, get_system_status
from src.pipeline import run_pipeline, run_quick_update
from src.regime import add_regime_features
from src.snr import compute_snr_levels, merge_multitimeframe_levels
from src.paper_trade_okx import execute_latest_signal_okx
from src.trade_journal import append_okx_order_record, load_okx_order_history
from src.walkforward import run_walkforward_validation, save_walkforward_report
from src.macro_events import generate_estimated_macro_events

INTERVAL_TO_SECONDS = {
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "1d": 24 * 60 * 60,
}

# ── 風格設定（僅影響槓桿偏好係數，不鎖 AI 信號門檻） ──────────────────
RISK_PROFILES = {
    "保守 🛡️": {"lev_mult": 0.40, "label": "保守", "color": "#38bdf8"},
    "中立 ⚖️": {"lev_mult": 0.70, "label": "中立", "color": "#a78bfa"},
    "激進 🔥": {"lev_mult": 1.00, "label": "激進", "color": "#f97316"},
}


def _safe_read_csv(path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        fallback = dict(kwargs)
        fallback.setdefault("engine", "python")
        fallback.setdefault("on_bad_lines", "skip")
        return pd.read_csv(path, **fallback)


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Arrow 安全層：將所有 object 欄位轉為 str，
    避免含 % 等字串的欄位被 pyarrow 試圖轉為 double 導致 ArrowTypeError。
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str)
    return out

def _to_utc_timestamp(value: object) -> pd.Timestamp | pd.NaT:
    try:
        ts = pd.to_datetime(value, utc=True)
    except Exception:
        return pd.NaT
    return ts


def _format_ts_dual(value: object) -> tuple[str, str]:
    ts = _to_utc_timestamp(value)
    if pd.isna(ts):
        return "N/A", "N/A"
    utc_text = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    tw_text = ts.tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S UTC+8")
    return utc_text, tw_text


def _format_tw(value: object) -> str:
    ts = _to_utc_timestamp(value)
    if pd.isna(ts):
        return "N/A"
    return ts.tz_convert("Asia/Taipei").strftime("%m/%d %H:%M")


def _bar_close_time_from_open(value: object, interval: str) -> pd.Timestamp | pd.NaT:
    ts = _to_utc_timestamp(value)
    if pd.isna(ts):
        return pd.NaT
    sec = int(INTERVAL_TO_SECONDS.get(interval, 0))
    if sec <= 0:
        return pd.NaT
    return ts + pd.Timedelta(seconds=sec - 1)


def _infer_interval_seconds_from_signals(df: pd.DataFrame) -> int:
    if "timestamp" not in df.columns or len(df) < 3:
        return 0
    x = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if len(x) < 3:
        return 0
    diffs = x.diff().dt.total_seconds().dropna()
    if diffs.empty:
        return 0
    return int(diffs.mode().iloc[0])


def _should_run_quick_update_now(df: pd.DataFrame, interval: str) -> bool:
    sec = int(INTERVAL_TO_SECONDS.get(interval, 0))
    if sec <= 0:
        return True
    if df.empty or "timestamp" not in df.columns:
        return True
    last_ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True, errors="coerce")
    if pd.isna(last_ts):
        return True
    now_utc = pd.Timestamp.now(tz="UTC")
    due_time = last_ts + pd.Timedelta(seconds=sec + 5)
    return now_utc >= due_time


def _direction_confidence(p_long: float, p_short: float, p_flat: float) -> float:
    """Directional confidence in [0, 1], avoids being stuck at 0 when p_flat is high."""
    try:
        pl = float(p_long)
        ps = float(p_short)
        pf = float(p_flat)
    except Exception:
        return 0.0
    vals = [0.0 if pd.isna(v) else max(0.0, min(1.0, v)) for v in (pl, ps, pf)]
    return max(vals[0], vals[1])


def _ai_classify_style(row: pd.Series) -> tuple[str, str, float]:
    """
    AI 自動判斷市場風格。
    回傳 (style_label, style_key, style_score)
    style_score: 連續評分 -3.0 (極保守) ~ +3.0 (極激進)
    """
    fg = float(row.get("fear_greed_value", 50) or 50)
    vol24 = float(row.get("realized_vol_24", 0.03) or 0.03)
    atr_pct = float(row.get("atr_pct", 0.015) or 0.015)
    p_long = float(row.get("p_long", 0.33) or 0.33)
    p_short = float(row.get("p_short", 0.33) or 0.33)
    p_flat = float(row.get("p_flat", 0.34) or 0.34)
    macd_hist = float(row.get("macd_hist", 0) or 0)
    drawdown = float(row.get("drawdown", 0) or 0)

    confidence = _direction_confidence(p_long, p_short, p_flat)

    score = 0.0

    # 恐懼貪婪因子
    if fg >= 75:
        score += 1.2   # 極度貪婪→積極
    elif fg >= 55:
        score += 0.6
    elif fg <= 25:
        score -= 1.5   # 極度恐懼→保守
    elif fg <= 40:
        score -= 0.7

    # 波動率因子（低波動可以更積極）
    if vol24 < 0.015:
        score += 0.8
    elif vol24 < 0.025:
        score += 0.3
    elif vol24 > 0.06:
        score -= 1.2
    elif vol24 > 0.04:
        score -= 0.6

    # ATR 相對波動
    if atr_pct < 0.008:
        score += 0.5
    elif atr_pct > 0.025:
        score -= 0.8

    # 信號信心度
    if confidence >= 0.35:
        score += 0.8
    elif confidence >= 0.20:
        score += 0.3
    elif confidence < 0.05:
        score -= 0.5

    # MACD 方向性
    if abs(macd_hist) > 0:
        score += 0.4 * (1 if macd_hist > 0 else -1)

    # 回撤懲罰
    if drawdown < -0.15:
        score -= 1.0
    elif drawdown < -0.08:
        score -= 0.5

    score = max(-3.0, min(3.0, score))

    if score >= 0.8:
        return "激進 🔥", "激進 🔥", score
    elif score <= -0.6:
        return "保守 🛡️", "保守 🛡️", score
    else:
        return "中立 ⚖️", "中立 ⚖️", score


def _build_bull_bear_reasons(row: pd.Series) -> tuple[list[str], list[str]]:
    p_long = float(row.get("p_long", 0.33) or 0.33)
    p_short = float(row.get("p_short", 0.33) or 0.33)
    p_flat = float(row.get("p_flat", 0.34) or 0.34)
    macd_hist = float(row.get("macd_hist", 0.0) or 0.0)
    rsi = float(row.get("rsi_14", 50.0) or 50.0)
    fear_greed = float(row.get("fear_greed_value", 50.0) or 50.0)
    vol24 = float(row.get("realized_vol_24", 0.03) or 0.03)
    atr_pct = float(row.get("atr_pct", 0.015) or 0.015)
    drawdown = float(row.get("drawdown", 0.0) or 0.0)

    bull: list[str] = []
    bear: list[str] = []

    if p_long > p_short:
        bull.append(f"看漲機率高於看跌（{p_long*100:.1f}% > {p_short*100:.1f}%）。")
    elif p_short > p_long:
        bear.append(f"看跌機率高於看漲（{p_short*100:.1f}% > {p_long*100:.1f}%）。")
    if p_flat >= 0.45:
        bear.append(f"觀望機率偏高（{p_flat*100:.1f}%），代表市場方向不明。")

    if macd_hist > 0:
        bull.append("MACD 柱體為正，動能偏多。")
    elif macd_hist < 0:
        bear.append("MACD 柱體為負，動能偏空。")

    if rsi <= 35:
        bull.append(f"RSI 偏低（{rsi:.1f}），存在反彈機會。")
    elif rsi >= 65:
        bear.append(f"RSI 偏高（{rsi:.1f}），短線回落風險上升。")

    if fear_greed >= 70:
        bull.append(f"恐懼貪婪指數偏高（{fear_greed:.0f}），市場情緒偏多。")
    elif fear_greed <= 30:
        bear.append(f"恐懼貪婪指數偏低（{fear_greed:.0f}），風險偏好不足。")

    if vol24 > 0.05 or atr_pct > 0.02:
        bear.append("波動率偏高，假突破與回撤風險增加。")
    elif vol24 < 0.025 and atr_pct < 0.012:
        bull.append("波動率相對可控，趨勢延續機率較佳。")

    if drawdown <= -0.10:
        bear.append("近期回撤偏深，模型風控會傾向保守。")

    if not bull:
        bull.append("目前偏多依據不足，需等待更明確的突破訊號。")
    if not bear:
        bear.append("目前偏空依據不足，空方動能尚未明顯擴大。")

    return bull[:4], bear[:4]


週期資料門檻 = {
    "5m": 200_000,
    "15m": 80_000,
    "30m": 70_000,
    "1h": 40_000,
    "1d": 3_000,
}


def _interval_seconds(interval: str) -> int:
    return int(
        {
            "5m": 5 * 60,
            "15m": 15 * 60,
            "30m": 30 * 60,
            "1h": 60 * 60,
            "1d": 24 * 60 * 60,
        }.get(interval, 60 * 60)
    )


def _expectancy_unit(win_rate: float, pnl_ratio: float) -> float:
    """
    Expectancy in loss-unit space.
    > 0 means long-run positive expectancy.
    """
    w = max(0.0, min(1.0, float(win_rate)))
    r = max(0.0, float(pnl_ratio))
    return (w * r) - (1.0 - w)


def _validation_thresholds(interval: str) -> dict[str, float]:
    base_rows = int(週期資料門檻.get(interval, 40_000))
    min_days = (base_rows * _interval_seconds(interval)) / 86400.0
    base = {
        "5m": {"min_days": min_days, "min_trades": 120, "min_rows": base_rows},
        "15m": {"min_days": min_days, "min_trades": 100, "min_rows": base_rows},
        "30m": {"min_days": min_days, "min_trades": 80, "min_rows": base_rows},
        "1h": {"min_days": min_days, "min_trades": 60, "min_rows": base_rows},
        "1d": {"min_days": min_days, "min_trades": 30, "min_rows": base_rows},
    }
    return base.get(interval, {"min_days": min_days, "min_trades": 60, "min_rows": base_rows})


def _build_backtest_warnings(
    interval: str,
    sample_rows: int,
    sample_days: float,
    bt: dict,
    wf_report: dict | None,
) -> tuple[str, list[str]]:
    limits = _validation_thresholds(interval)
    warnings: list[str] = []
    severity = "ok"

    trades = int(bt.get("trades", 0) or 0)
    if sample_rows < int(limits["min_rows"]):
        warnings.append(f"回測K線數只有 {sample_rows:,} 根，低於建議的 {int(limits['min_rows']):,} 根。")
        severity = "warning"
    if sample_days < float(limits["min_days"]):
        warnings.append(f"回測期間約 {sample_days:,.1f} 天，低於此週期建議的 {float(limits['min_days']):,.0f} 天。")
        severity = "warning"
    if trades < int(limits["min_trades"]):
        warnings.append(f"交易筆數只有 {trades} 筆，低於建議的 {int(limits['min_trades'])} 筆。")
        severity = "warning"

    wr = float(bt.get("win_rate", 0.0) or 0.0)
    rr = float(bt.get("pnl_ratio", 0.0) or 0.0)
    exp_u = _expectancy_unit(wr, rr)
    if exp_u <= 0:
        warnings.append(f"主回測期望值 <= 0（勝率 {wr*100:.2f}%、盈虧比 {rr:.3f}、期望值 {exp_u:.4f}）。")
        severity = "critical"

    if wf_report:
        summary = wf_report.get("summary", {}) if isinstance(wf_report.get("summary"), dict) else {}
        positive_expectancy_folds = int(summary.get("positive_expectancy_folds", 0) or 0)
        fold_count = int(wf_report.get("fold_count", 0) or 0)
        avg_fold_expectancy = float(summary.get("average_fold_expectancy_unit", 0.0) or 0.0)
        if fold_count > 0 and positive_expectancy_folds < max(1, fold_count // 2):
            warnings.append(f"Walk-forward 只有 {positive_expectancy_folds}/{fold_count} 個 fold 為正期望值。")
            severity = "critical"
        if avg_fold_expectancy <= 0:
            warnings.append(f"Walk-forward 平均期望值 <= 0（{avg_fold_expectancy:.4f}），代表泛化能力不足。")
            severity = "critical"

    if not warnings:
        return "ok", ["樣本數、交易筆數與 walk-forward 目前都在可接受範圍內。"]
    return severity, warnings


def _is_walkforward_stale(wf_report: dict | None, current_rows: int, current_end_utc: str) -> bool:
    if not wf_report:
        return True
    try:
        src_rows = int(wf_report.get("source_rows", 0) or 0)
    except Exception:
        return True
    src_end = str(wf_report.get("source_end_utc", "") or "")
    return (src_rows != int(current_rows)) or (src_end != str(current_end_utc))


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
UI_PREFS_PATH = OUTPUT_DIR / "dashboard_user_prefs.json"
BALANCE_AUTO_REFRESH_SEC = 30
預設交易對 = "BTCUSDT"
多週期清單 = ["5m", "15m", "30m", "1h", "1d"]
每週期K線顯示保底 = 5000


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

st.set_page_config(
    page_title="BTC AI 智能交易儀表板",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
      @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:FILL@0;wght@400;GRAD@0;opsz@24');
      @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
      .stApp, .stApp * {
        font-family: 'Inter', sans-serif !important;
      }
      /* Keep Streamlit/Material icons from being replaced by Inter text glyphs */
      [data-testid="stIconMaterial"],
      .material-icons,
      .material-icons-outlined,
      .material-icons-round,
      .material-icons-sharp,
      .material-icons-two-tone,
      .material-symbols-outlined,
      .material-symbols-rounded,
      .material-symbols-sharp,
      [class^="material-symbols"],
      [class*=" material-symbols"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Symbols Sharp", "Material Icons" !important;
        font-style: normal !important;
        font-weight: 400 !important;
        line-height: 1 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        -webkit-font-smoothing: antialiased !important;
        font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24 !important;
      }
      .stApp {
        background: radial-gradient(ellipse at 15% 0%, #0f1729 0%, #020810 55%, #0a0f1e 100%);
        color: #f0f4ff;
      }
      section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0b1120 100%);
        border-right: 1px solid #1e2d45;
      }
      .metric-title { font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }
      .metric-value { font-size: 2.3rem; font-weight: 800; line-height: 1.1; }
      .signal-line { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em; }
      .signal-bull { color: #22c55e; }
      .signal-bear { color: #ef4444; }
      .signal-flat { color: #facc15; }
      .subtle { color: #64748b; font-size: 0.9rem; margin-top: 2px; }

      /* AI 風格卡片 */
      .style-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px; padding: 16px 20px; margin: 8px 0;
        border: 1px solid #334155;
      }
      .style-badge {
        display: inline-block; border-radius: 999px;
        padding: 4px 14px; font-size: 1.1rem; font-weight: 700;
        margin-bottom: 4px;
      }
      .style-aggressive { background: linear-gradient(90deg,#f97316,#ea580c); color:#fff; }
      .style-neutral    { background: linear-gradient(90deg,#a78bfa,#7c3aed); color:#fff; }
      .style-conservative { background: linear-gradient(90deg,#38bdf8,#0284c7); color:#fff; }

      /* Metric cards */
      .metric-card {
        background: linear-gradient(135deg,#1e293b 0%,#0f1829 100%);
        border: 1px solid #1e3a5f; border-radius: 14px;
        padding: 18px 20px; margin-bottom: 4px;
      }

      /* Table styling */
      .stDataFrame { border-radius: 12px; overflow: hidden; }

      /* Confidence bar */
      .conf-bar-bg { background:#1e293b; border-radius:999px; height:8px; }
      .conf-bar-fill { border-radius:999px; height:8px; }

      /* Keep cursor style unchanged over Plotly charts */
      .js-plotly-plot, .js-plotly-plot * {
        cursor: default !important;
      }

      /* Guard against sporadic duplicated tab headers in Streamlit frontend */
      [data-testid="stTabs"] [data-baseweb="tab-list"] + [data-baseweb="tab-list"] {
        display: none !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def 讀取訊號資料() -> pd.DataFrame:
    if not 目前訊號檔.exists():
        舊檔 = OUTPUT_DIR / "signals_with_features.csv"
        if 週期 == "1h" and 舊檔.exists():
            df = _safe_read_csv(舊檔)
        else:
            return pd.DataFrame()
    else:
        df = _safe_read_csv(目前訊號檔)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "regime" not in df.columns:
        try:
            df = add_regime_features(df)
        except Exception:
            pass
    return df.sort_values("timestamp").reset_index(drop=True)


def 讀取報告() -> dict:
    if 目前報告檔.exists():
        return json.loads(目前報告檔.read_text(encoding="utf-8"))
    舊檔 = OUTPUT_DIR / "report.json"
    if 舊檔.exists():
        return json.loads(舊檔.read_text(encoding="utf-8"))
    return {}


def 讀取交易明細(symbol: str, interval: str) -> pd.DataFrame:
    tag3 = f"{symbol}_{interval}"
    p = OUTPUT_DIR / f"trades_{tag3}.csv"
    if not p.exists():
        p = OUTPUT_DIR / "trades.csv"
    if not p.exists():
        return pd.DataFrame()
    return _safe_read_csv(p)


def 取得數值(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def 百分比(x: float) -> str:
    return f"{x * 100:.2f}%"


def 回測顯示值(指標: str, value: object) -> object:
    if value is None:
        return value
    try:
        v = float(value)
    except Exception:
        return value
    if 指標 in {"總收益", "最大回撤", "勝率", "VaR 95%", "ES 95%"}:
        return f"{v * 100:.2f}%"
    return v


def 判斷訊號(p_long: float, p_short: float, 門檻: float) -> tuple[str, str, str]:
    if p_long >= 門檻 and p_long > p_short:
        return "看漲", "買入 / 做多", "signal-bull"
    if p_short >= 門檻 and p_short > p_long:
        return "看跌", "賣出 / 做空", "signal-bear"
    return "觀望", "等待", "signal-flat"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return f"rgba(255,255,255,{alpha:.2f})"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _snr_style(overlap_count: int, kind: str) -> tuple[str, float, str, int]:
    """
    S and non-S use different colors.
    Overlap count controls opacity from 0.3 to 1.0 and line width from 1px to 4px.
    """
    count = max(1, min(4, int(overlap_count)))
    opacity = 0.3 + ((count - 1) / 3.0) * 0.7
    line_width = count
    base_color = "#60a5fa" if kind == "S" else "#fb7185"
    return base_color, opacity, _hex_to_rgba(base_color, max(0.18, opacity * 0.35)), line_width


def K線圖(df: pd.DataFrame) -> go.Figure:
    x = df.copy().sort_values("timestamp").reset_index(drop=True)
    y_low = float(x["low"].min())
    y_high = float(x["high"].max())
    y_span = max(y_high - y_low, float(x["close"].iloc[-1]) * 0.001)
    y_pad = y_span * 0.12

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=x["timestamp"],
            open=x["open"],
            high=x["high"],
            low=x["low"],
            close=x["close"],
            name="BTCUSDT",
            increasing_line_color="#86efac",
            decreasing_line_color="#fca5a5",
            increasing_fillcolor="#166534",
            decreasing_fillcolor="#7f1d1d",
            hovertemplate=(
                "時間: %{x|%Y-%m-%d %H:%M:%S}<br>"
                "開: %{open:,.2f}<br>"
                "高: %{high:,.2f}<br>"
                "低: %{low:,.2f}<br>"
                "收: %{close:,.2f}<extra></extra>"
            ),
        )
    )

    if "signal" in x.columns:
        sig = pd.to_numeric(x["signal"], errors="coerce").fillna(0)
        prev_sig = sig.shift(1).fillna(0)
        buy_mask = (sig == 1) & (prev_sig != 1)
        sell_mask = (sig == -1) & (prev_sig != -1)

        buy_df = x[buy_mask]
        sell_df = x[sell_mask]

        if not buy_df.empty:
            buy_conf = (pd.to_numeric(buy_df.get("confidence_index", 0), errors="coerce").fillna(0) * 100).round(1)
            fig.add_trace(
                go.Scatter(
                    x=buy_df["timestamp"],
                    y=buy_df["low"] * 0.998,
                    mode="markers+text",
                    name="買點",
                    marker=dict(symbol="triangle-up", size=13, color="#22c55e",
                                line=dict(color="#166534", width=1)),
                    text=[f"▲ {v:.0f}%" for v in buy_conf],
                    textposition="top center",
                    textfont=dict(color="#22c55e", size=10),
                    customdata=buy_conf,
                    hovertemplate=(
                        "時間: %{x|%Y-%m-%d %H:%M:%S}<br>"
                        "買點信心: %{customdata:.1f}%<extra></extra>"
                    ),
                )
            )
        if not sell_df.empty:
            sell_conf = (pd.to_numeric(sell_df.get("confidence_index", 0), errors="coerce").fillna(0) * 100).round(1)
            fig.add_trace(
                go.Scatter(
                    x=sell_df["timestamp"],
                    y=sell_df["high"] * 1.002,
                    mode="markers+text",
                    name="賣點",
                    marker=dict(symbol="triangle-down", size=13, color="#ef4444",
                                line=dict(color="#7f1d1d", width=1)),
                    text=[f"▼ {v:.0f}%" for v in sell_conf],
                    textposition="bottom center",
                    textfont=dict(color="#ef4444", size=10),
                    customdata=sell_conf,
                    hovertemplate=(
                        "時間: %{x|%Y-%m-%d %H:%M:%S}<br>"
                        "賣點信心: %{customdata:.1f}%<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.02, x=0),
        uirevision="kline-static",
        dragmode="pan",
        hovermode="x unified",
        hoverdistance=120,
        spikedistance=1000,
    )
    fig.update_yaxes(
        title="價格 (USDT)",
        range=[y_low - y_pad, y_high + y_pad],
        autorange=False,
        gridcolor="#1e293b",
    )
    fig.update_xaxes(
        gridcolor="#1e293b",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="#64748b",
    )
    return fig


def 買賣橫條圖(p_long: float, p_short: float, p_flat: float) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=[p_long * 100, p_short * 100, p_flat * 100],
                y=["看漲機率", "看跌機率", "觀望機率"],
                orientation="h",
                marker=dict(
                    color=["#22c55e", "#ef4444", "#facc15"],
                    line=dict(width=0),
                ),
                text=[f"{p_long*100:.2f}%", f"{p_short*100:.2f}%", f"{p_flat*100:.2f}%"],
                textposition="outside",
                textfont=dict(size=14, color="#f0f4ff"),
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        height=240,
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
        xaxis_title="機率 (%)",
        xaxis=dict(gridcolor="#1e293b", range=[0, 105]),
    )
    return fig


def 趨勢高低點圖(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="收盤價",
                              line=dict(color="#e2e8f0", width=2)))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["rolling_high_24"], name="24h高點",
                              line=dict(color="#22c55e", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["rolling_low_24"], name="24h低點",
                              line=dict(color="#ef4444", width=1.5, dash="dot")))
    fig.update_layout(
        template="plotly_dark", height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
        uirevision="trend-static",
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
    )
    return fig


def MACD圖(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["macd_hist"], name="MACD柱",
                          marker_color=colors, opacity=0.7))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["macd"], name="MACD",
                              line=dict(color="#0ea5e9", width=2)))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["macd_signal"], name="訊號線",
                              line=dict(color="#f59e0b", width=2)))
    fig.update_layout(
        template="plotly_dark", height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
        uirevision="macd-static",
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
    )
    return fig


def ATR圖(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    atr14 = pd.to_numeric(df.get("atr_14"), errors="coerce")
    atr_pct = pd.to_numeric(df.get("atr_pct"), errors="coerce") * 100.0

    fig.add_trace(go.Scatter(x=df["timestamp"], y=atr14, name="ATR(14)",
                              line=dict(color="#a78bfa", width=2)))
    # If ATR% is effectively all zeros, hide the flat zero line to reduce confusion.
    if atr_pct.notna().any() and float(atr_pct.fillna(0).abs().max()) > 1e-9:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=atr_pct, name="ATR%",
                                  line=dict(color="#f43f5e", width=2)))
    fig.update_layout(
        template="plotly_dark", height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
        yaxis_title="ATR / ATR%", uirevision="atr-static",
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
    )
    return fig


def 恐懼貪婪儀表(value: float) -> go.Figure:
    # Color based on value
    if value <= 25:
        bar_color = "#ef4444"
    elif value <= 45:
        bar_color = "#f97316"
    elif value <= 55:
        bar_color = "#facc15"
    elif value <= 75:
        bar_color = "#84cc16"
    else:
        bar_color = "#22c55e"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            number={"suffix": " / 100", "font": {"size": 40}},
            title={"text": "恐懼與貪婪指數", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                "bar": {"color": bar_color, "thickness": 0.25},
                "bgcolor": "#0f172a",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#7f1d1d"},
                    {"range": [25, 45], "color": "#78350f"},
                    {"range": [45, 55], "color": "#1c1917"},
                    {"range": [55, 75], "color": "#14532d"},
                    {"range": [75, 100], "color": "#064e3b"},
                ],
                "threshold": {"line": {"color": "#f0f4ff", "width": 3}, "thickness": 0.8, "value": value},
            },
        )
    )
    fig.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def 事件清單(row: pd.Series) -> list[str]:
    items: list[str] = []
    if 取得數值(row, "etf_news_score") > 0:
        items.append("ETF 相關事件")
    if 取得數值(row, "regulatory_news_score") > 0:
        items.append("監管/法規事件")
    if 取得數值(row, "exchange_event_score") > 0:
        items.append("交易所事件")
    if 取得數值(row, "black_swan_risk_score") > 0:
        items.append("黑天鵝風險事件")
    if int(取得數值(row, "news_shock", 0)) == 1:
        items.append("新聞衝擊事件")
    if 取得數值(row, "fed_news_score") > 0:
        items.append("美聯儲/FOMC 相關新聞")
    if 取得數值(row, "trump_news_score") > 0:
        items.append("川普相關新聞")
    if 取得數值(row, "war_news_score") > 0:
        items.append("戰爭/地緣衝突風險新聞")
    if 取得數值(row, "panic_news_score") > 0:
        items.append("金融市場恐慌訊號")
    if 取得數值(row, "macro_event_risk_score") > 0:
        items.append("CPI/PPI/FOMC 公布時段（風險降槓桿）")
    if not items:
        items.append("目前無明顯事件訊號")
    return items


def _build_past_event_table(signals_df: pd.DataFrame, max_rows: int = 40) -> pd.DataFrame:
    if signals_df.empty:
        return pd.DataFrame(columns=["時間(台北)", "事件", "風險分數"])
    s = signals_df.copy().sort_values("timestamp")
    def _num_col(df: pd.DataFrame, name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0)
        return pd.Series(0.0, index=df.index, dtype="float64")
    mask = (
        (_num_col(s, "war_news_score") > 0)
        | (_num_col(s, "panic_news_score") > 0)
        | (_num_col(s, "fed_news_score") > 0)
        | (_num_col(s, "trump_news_score") > 0)
        | (_num_col(s, "macro_event_risk_score") > 0)
        | (_num_col(s, "black_swan_risk_score") > 0)
    )
    e = s[mask].tail(max_rows).copy()
    if e.empty:
        return pd.DataFrame(columns=["時間(台北)", "事件", "風險分數"])
    out = pd.DataFrame()
    out["時間(台北)"] = pd.to_datetime(e["timestamp"], utc=True).dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
    out["事件"] = e.apply(lambda r: "、".join(事件清單(r)), axis=1)
    out["風險分數"] = pd.to_numeric(e.get("market_panic_score", 0), errors="coerce").fillna(0).round(2)
    return out.sort_values("時間(台北)", ascending=False).reset_index(drop=True)


def _build_future_event_table(now_utc: pd.Timestamp, days: int = 120) -> pd.DataFrame:
    start = pd.to_datetime(now_utc, utc=True)
    end = start + pd.Timedelta(days=days)
    ev = generate_estimated_macro_events(start, end)
    if ev.empty:
        return pd.DataFrame(columns=["時間(台北)", "事件", "類型", "風險權重"])
    out = pd.DataFrame()
    out["時間(台北)"] = pd.to_datetime(ev["timestamp"], utc=True).dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
    out["事件"] = ev["event_name"].astype(str)
    out["類型"] = ev["event_type"].astype(str)
    out["風險權重"] = pd.to_numeric(ev["risk_weight"], errors="coerce").fillna(0).round(2)
    return out.reset_index(drop=True)


def 同步槓桿設定(prefix: str, label: str, default: int, min_value: int = 1, max_value: int = 100) -> int:
    slider_key = f"{prefix}_slider"
    input_key = f"{prefix}_input"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = int(default)
    if input_key not in st.session_state:
        st.session_state[input_key] = int(default)

    def _from_slider() -> None:
        st.session_state[input_key] = int(st.session_state[slider_key])

    def _from_input() -> None:
        v = int(st.session_state[input_key])
        v = max(min_value, min(max_value, v))
        st.session_state[input_key] = v
        st.session_state[slider_key] = v

    st.sidebar.markdown(label)
    c1, c2 = st.sidebar.columns([3, 1])
    with c1:
        st.slider(
            f"{label} 滑桿",
            min_value=min_value,
            max_value=max_value,
            key=slider_key,
            label_visibility="collapsed",
            on_change=_from_slider,
        )
    with c2:
        st.number_input(
            f"{label} 輸入",
            min_value=min_value,
            max_value=max_value,
            step=1,
            key=input_key,
            label_visibility="collapsed",
            on_change=_from_input,
        )
    return int(st.session_state[input_key])


# ─── 新：盈虧折線圖 ────────────────────────────────────────────────────────
def 盈虧折線圖(trades: pd.DataFrame) -> go.Figure:
    df = trades.copy()
    # 找出策略報酬欄位
    ret_col = "策略報酬(含費用)" if "策略報酬(含費用)" in df.columns else None
    if ret_col is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text="尚無交易資料", font=dict(color="#64748b")),
        )
        return fig

    # 使用出場時間做 X 軸；若有 pnl_usdt 就用，否則用報酬率累計
    if "pnl_usdt" in df.columns:
        df["cum_pnl"] = df["pnl_usdt"].fillna(0).cumsum()
        y_label = "累計盈虧 (USDT)"
    else:
        df["cum_pnl"] = ((1 + df[ret_col].fillna(0)).cumprod() - 1) * 100
        y_label = "累計報酬 (%)"

    x_col = "出場時間" if "出場時間" in df.columns else df.columns[0]
    x_vals = pd.to_datetime(df[x_col], errors="coerce")

    colors_area = ["#22c55e" if v >= 0 else "#ef4444" for v in df["cum_pnl"]]
    last_val = float(df["cum_pnl"].iloc[-1]) if not df.empty else 0
    line_color = "#22c55e" if last_val >= 0 else "#ef4444"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=df["cum_pnl"],
        fill="tozeroy",
        fillcolor="rgba(34,197,94,0.12)" if last_val >= 0 else "rgba(239,68,68,0.12)",
        line=dict(color=line_color, width=2.5),
        mode="lines",
        name=y_label,
        hovertemplate="%{x|%m/%d %H:%M}<br>" + y_label + ": %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#475569", line_width=1, line_dash="dot")
    fig.update_layout(
        template="plotly_dark", height=340,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
        yaxis_title=y_label, xaxis_title="出場時間",
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        title=dict(text=f"總累計 {last_val:+.4f}", font=dict(color=line_color, size=15), x=0.98, xanchor="right"),
    )
    return fig


# ─── 新：交易明細表格（補充 p_long / p_short / 信心指數） ──────────────────
def 格式化交易明細(trades: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    df = trades.copy()

    # 嘗試補充 p_long / p_short / confidence / ai_style
    if not signals.empty and "timestamp" in signals.columns:
        sig_lookup = signals.set_index("timestamp")

        def _match_num(ts_str: str, col: str, default: float = 0.0) -> float:
            """從 signals 查詢數值欄位。"""
            try:
                ts = pd.to_datetime(ts_str, utc=True)
                if ts in sig_lookup.index:
                    return float(sig_lookup.loc[ts, col])
                idx = sig_lookup.index.get_indexer([ts], method="nearest")[0]
                return float(sig_lookup.iloc[idx][col])
            except Exception:
                return default

        def _match_str(ts_str: str, col: str, default: str = "") -> str:
            """從 signals 查詢字串欄位。"""
            try:
                ts = pd.to_datetime(ts_str, utc=True)
                if ts in sig_lookup.index:
                    return str(sig_lookup.loc[ts, col])
                idx = sig_lookup.index.get_indexer([ts], method="nearest")[0]
                return str(sig_lookup.iloc[idx][col])
            except Exception:
                return default

        if "p_long" in signals.columns:
            df["看漲機率"] = df["進場時間"].apply(lambda t: f"{_match_num(t,'p_long')*100:.1f}%")
        if "p_short" in signals.columns:
            df["看跌機率"] = df["進場時間"].apply(lambda t: f"{_match_num(t,'p_short')*100:.1f}%")
        if all(c in signals.columns for c in ["p_long", "p_short", "p_flat"]):
            df["信心指數"] = df["進場時間"].apply(
                lambda t: f"{_direction_confidence(_match_num(t,'p_long'), _match_num(t,'p_short'), _match_num(t,'p_flat',0.34)) * 100:.1f}%"
            )
        if "ai_style" in signals.columns:
            df["AI風格"] = df["進場時間"].apply(lambda t: _match_str(t, "ai_style", "中立"))

    # 格式化時間欄位
    for col in ["進場時間", "出場時間"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _format_tw(v))

    # 盈虧 USDT 估算
    if "pnl_usdt" not in df.columns and "策略報酬(含費用)" in df.columns and "進場價" in df.columns:
        notional = 50.0
        lev_col = pd.to_numeric(df.get("槓桿", pd.Series([1.0] * len(df))), errors="coerce").fillna(1.0)
        df["盈虧(USDT)"] = (
            pd.to_numeric(df["策略報酬(含費用)"], errors="coerce").fillna(0) * notional * lev_col
        ).round(3)
    elif "pnl_usdt" in df.columns:
        df["盈虧(USDT)"] = pd.to_numeric(df["pnl_usdt"], errors="coerce").fillna(0).round(3)

    if "策略報酬(含費用)" in df.columns:
        df["盈虧%"] = (
            pd.to_numeric(df["策略報酬(含費用)"], errors="coerce").fillna(0) * 100
        ).round(3).astype(str) + "%"

    # 確保數値欄位是 Arrow 安全的形態
    if "槓桿" in df.columns:
        df["槓桿"] = pd.to_numeric(df["槓桿"], errors="coerce").round(2)
    for pcol in ["進場價", "出場價"]:
        if pcol in df.columns:
            df[pcol] = pd.to_numeric(df[pcol], errors="coerce").round(2)
    if "持倉K數" in df.columns:
        df["持倉K數"] = pd.to_numeric(df["持倉K數"], errors="coerce").astype("Int64")

    wanted = ["進場時間", "進場價", "方向", "槓桿", "出場時間", "出場價",
              "看漲機率", "看跌機率", "信心指數", "AI風格", "盈虧(USDT)", "盈虧%", "持倉K數"]
    show_cols = [c for c in wanted if c in df.columns]
    return df[show_cols]


def _嘗試載入週期資料(symbol: str, interval: str) -> pd.DataFrame | None:
    tag2 = f"{symbol}_{interval}"
    p_outputs = OUTPUT_DIR / f"signals_with_features_{tag2}.csv"
    p_raw = BASE_DIR / "data" / f"{symbol}_{interval}_ohlcv.csv"
    if p_outputs.exists():
        d = pd.read_csv(p_outputs)
    elif p_raw.exists():
        d = pd.read_csv(p_raw)
    else:
        return None
    if "timestamp" in d.columns:
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    return d.sort_values("timestamp").reset_index(drop=True)


# ═══════════════════ SIDEBAR ═══════════════════════════════════════════════
st.sidebar.markdown("## ⚙️ 設定")
_ui_prefs = _read_json_file(UI_PREFS_PATH)

def _pref_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        v = int(_ui_prefs.get(name, default))
    except Exception:
        v = int(default)
    if min_value is not None:
        v = max(int(min_value), v)
    if max_value is not None:
        v = min(int(max_value), v)
    return v


def _pref_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        v = float(_ui_prefs.get(name, default))
    except Exception:
        v = float(default)
    if min_value is not None:
        v = max(float(min_value), v)
    if max_value is not None:
        v = min(float(max_value), v)
    return v


# Restore last-used UI controls once per new session.
if not bool(st.session_state.get("_ui_prefs_loaded_once", False)):
    st.session_state["ui_symbol"] = str(_ui_prefs.get("symbol", 預設交易對))
    st.session_state["ui_interval"] = str(_ui_prefs.get("kline_interval", "1h"))
    st.session_state["ui_kline_count"] = _pref_int("kline_count", 300, min_value=100, max_value=2000)
    st.session_state["ui_backtest_sample_rows"] = _pref_int("backtest_sample_rows", 0, min_value=0, max_value=2000000)
    st.session_state["ui_kline_auto_update_enabled"] = bool(_ui_prefs.get("kline_auto_update_enabled", False))
    st.session_state["ui_kline_auto_update_sec"] = _pref_int("kline_auto_update_sec", 15, min_value=5, max_value=3600)
    st.session_state["ui_max_train_rows"] = _pref_int("max_train_rows", 40000, min_value=0, max_value=2000000)
    st.session_state["ui_black_swan_reserve_usdt"] = _pref_float("black_swan_reserve_usdt", 0.0, min_value=0.0, max_value=10_000_000.0)
    st.session_state["ui_black_swan_threshold"] = _pref_float("black_swan_threshold", 1.0, min_value=0.0, max_value=10.0)
    st.session_state["_ui_prefs_loaded_once"] = True

交易對 = st.sidebar.text_input("交易對", value=st.session_state["ui_symbol"], key="ui_symbol")
if str(st.session_state.get("ui_interval", "1h")) not in 多週期清單:
    st.session_state["ui_interval"] = "1h"
週期 = st.sidebar.selectbox("K線週期", 多週期清單, key="ui_interval")
if "_last_interval_selected" not in st.session_state:
    st.session_state["_last_interval_selected"] = 週期
if st.session_state["_last_interval_selected"] != 週期:
    st.session_state["_last_interval_selected"] = 週期
    st.session_state["_sync_after_interval_switch"] = True
槓桿上限 = 同步槓桿設定("槓桿上限", "槓桿上限 (1~100)", default=100)
K線根數 = st.sidebar.slider("K線顯示根數", min_value=100, max_value=2000, step=50, key="ui_kline_count")
回測樣本數 = st.sidebar.number_input("回測摘要樣本數 (0=全部可用)", min_value=0, max_value=2_000_000, step=500, key="ui_backtest_sample_rows")

st.sidebar.divider()
st.sidebar.markdown("### 🎯 風險偏好")
st.sidebar.caption("AI 自動判斷市場風格（激進/中立/保守）；此選項只作為槓桿倍率的偏好係數。")
if "risk_profile" not in st.session_state:
    _saved_risk = str(_ui_prefs.get("risk_profile", "中立 ⚖️"))
    if _saved_risk not in RISK_PROFILES:
        _saved_risk = list(RISK_PROFILES.keys())[1]
    st.session_state["risk_profile"] = _saved_risk
風險偏好 = st.sidebar.radio(
    "槓桿偏好",
    options=list(RISK_PROFILES.keys()),
    label_visibility="collapsed",
    key="risk_profile",
)
槓桿偏好係數 = RISK_PROFILES[風險偏好]["lev_mult"]

st.sidebar.divider()
st.sidebar.markdown("### 📐 SNR(支撐/壓力) 策略")
顯示SNR = st.sidebar.checkbox("在K線上顯示SNR水平線", value=True)
SNR候選週期 = ["5m", "15m", "1h", "1d"]
SNR重疊層數 = st.sidebar.selectbox("SNR重疊條件", options=[1, 2, 3, 4], index=1,
                                    format_func=lambda n: f"至少{n}個週期重疊")
SNR最大線數 = st.sidebar.slider("每週期最多線數", min_value=2, max_value=15, value=8)

tag = f"{交易對}_{週期}"
目前訊號檔 = OUTPUT_DIR / f"signals_with_features_{tag}.csv"
目前報告檔 = OUTPUT_DIR / f"report_{tag}.json"
目前交易檔 = OUTPUT_DIR / f"trades_{tag}.csv"

st.sidebar.divider()
st.sidebar.markdown("### 🔄 資料更新")
st.sidebar.caption("全週期執行（5m/15m/30m/1h/1d）。K 線圖顯示仍以輕量為主，但資料保留量會依訓練/回測樣本需求自動放大。")
st.sidebar.caption("目前資料門檻：5m=200k、15m=80k、30m=70k、1h=40k、1d=3k")
按鈕全量 = st.sidebar.button("全部週期抓資料+訓練", use_container_width=True)
按鈕快速 = st.sidebar.button("全部週期快速更新", use_container_width=True)
按鈕增量重訓 = st.sidebar.button("全部週期增量重訓", use_container_width=True)
即時更新啟用 = st.sidebar.checkbox(
    "K線即時更新",
    key="ui_kline_auto_update_enabled",
    help="開啟後會定時做『快速更新』並自動刷新頁面。",
)
即時更新秒數 = st.sidebar.number_input(
    "K線即時更新秒數",
    min_value=5,
    max_value=3600,
    step=1,
    key="ui_kline_auto_update_sec",
)
訓練最大樣本數 = st.sidebar.number_input(
    "重訓最大樣本數 (0=全量)",
    min_value=0,
    max_value=2000000,
    step=10000,
    key="ui_max_train_rows",
)

st.sidebar.divider()
st.sidebar.markdown("### 🧑‍🏫 知識蒸餾 (Teacher)")
st.sidebar.caption("用現有 CSV 訓練大型 Teacher 集成模型，產生高品質軟標籤。")
_teacher_csv = 目前訊號檔  # 預設用目前週期的訊號檔
_teacher_max_rows = st.sidebar.number_input(
    "Teacher 最大訓練筆數 (0=全量)", min_value=0, max_value=500000, value=0, step=10000,
    key="teacher_max_rows"
)
_teacher_temperature = st.sidebar.slider(
    "軟標籤溫度 T（越大越平滑）", min_value=1.0, max_value=5.0, value=2.0, step=0.5,
    key="teacher_temp"
)
_teacher_n_rf = st.sidebar.select_slider(
    "RF 樹數", options=[100, 200, 300, 500, 800], value=500, key="teacher_n_rf"
)
按鈕訓練Teacher = st.sidebar.button(
    "🧑‍🏫 訓練 Teacher 模型", use_container_width=True, key="btn_train_teacher"
)
from src.distillation import teacher_exists, load_teacher_report
_teacher_model_dir = BASE_DIR / "models"
_teacher_ok = teacher_exists(_teacher_model_dir, 交易對, 週期)
if _teacher_ok:
    st.sidebar.success("✅ Teacher 模型已存在")
else:
    st.sidebar.info("尚未訓練 Teacher，點擊上方按鈕開始。")
okx_inst = st.sidebar.text_input("OKX 合約 instId", value="BTC-USDT-SWAP")
okx_notional = st.sidebar.number_input("下單本金(USDT)", min_value=5.0, max_value=100000.0,
                                         value=50.0, step=5.0)
okx_enable = st.sidebar.checkbox("允許送出模擬盤下單(OKX_ENABLE_TRADING=1)", value=False)
okx_sync_before_order = st.sidebar.checkbox("下單前先快速同步資料", value=False)
if "ui_black_swan_reserve_usdt" not in st.session_state:
    st.session_state["ui_black_swan_reserve_usdt"] = _pref_float("black_swan_reserve_usdt", 0.0, min_value=0.0, max_value=10_000_000.0)
if "ui_black_swan_threshold" not in st.session_state:
    st.session_state["ui_black_swan_threshold"] = _pref_float("black_swan_threshold", 1.0, min_value=0.0, max_value=10.0)
黑天鵝保留資金 = float(st.session_state.get("ui_black_swan_reserve_usdt", 0.0))
黑天鵝觸發門檻 = float(st.session_state.get("ui_black_swan_threshold", 1.0))


# ── 帳戶餘額即時查詢（供頂部 banner 顯示） ────────────────────────
_bal_cache_key = "okx_balance_cache"
_bal_last_fetch_key = "okx_balance_last_fetch_ts"

def _fetch_and_show_balance() -> None:
    import os as _os
    _os.environ.setdefault("OKX_SIMULATED", "1")
    try:
        from src.exchange_okx import OKXClient, OKXCredentials
        _creds = OKXCredentials(
            api_key=_os.getenv("OKX_API_KEY", ""),
            secret_key=_os.getenv("OKX_API_SECRET", ""),
            passphrase=_os.getenv("OKX_API_PASSPHRASE", ""),
        )
        _cli = OKXClient(creds=_creds, simulated=True)
        _resp = _cli.get_balance("USDT")
        _details = (_resp.get("data") or [{}])[0].get("details") or []
        _usdt = next(
            (float(d.get("eq") or d.get("availBal") or 0)
             for d in _details if d.get("ccy") == "USDT"),
            None,
        )
        if _usdt is None:
            _total = str((_resp.get("data") or [{}])[0].get("totalEq") or "N/A")
            st.session_state[_bal_cache_key] = {"usdt": None, "totalEq": _total, "fetched_at_utc": _utc_now_iso()}
        else:
            st.session_state[_bal_cache_key] = {"usdt": _usdt, "totalEq": None, "fetched_at_utc": _utc_now_iso()}
        st.session_state[_bal_last_fetch_key] = time.time()
    except Exception as _be:
        st.session_state[_bal_cache_key] = {"error": str(_be), "fetched_at_utc": _utc_now_iso()}
        st.session_state[_bal_last_fetch_key] = time.time()

_last_bal_ts = float(st.session_state.get(_bal_last_fetch_key, 0.0))
if (time.time() - _last_bal_ts) >= BALANCE_AUTO_REFRESH_SEC:
    _fetch_and_show_balance()

_bal_data = st.session_state.get(_bal_cache_key, {})


if "ui_auto_trade_enabled" not in st.session_state:
    st.session_state["ui_auto_trade_enabled"] = bool(_ui_prefs.get("auto_trade_enabled", False))
if "ui_auto_trade_sec" not in st.session_state:
    st.session_state["ui_auto_trade_sec"] = int(_ui_prefs.get("auto_trade_sec", 30))
if "ui_tp_pct" not in st.session_state:
    st.session_state["ui_tp_pct"] = float(_ui_prefs.get("tp_pct", 1.5))
if "ui_sl_pct" not in st.session_state:
    st.session_state["ui_sl_pct"] = float(_ui_prefs.get("sl_pct", 1.0))
st.sidebar.markdown("### ⚡ 純AI自動交易")
st.sidebar.caption("純AI：只在目前這個 dashboard 頁面運行，關頁或斷線會停止。")
自動交易啟用 = st.sidebar.checkbox("啟用純AI自動交易", key="ui_auto_trade_enabled",
                                     help="開啟後會定時使用最新版AI訊號自動下單。")
自動交易秒數 = st.sidebar.number_input("自動交易檢查秒數", min_value=10, max_value=3600, step=5, key="ui_auto_trade_sec")
自動止盈百分比 = st.sidebar.number_input("自動止盈(%)", min_value=0.1, max_value=50.0, step=0.1, key="ui_tp_pct")
自動止損百分比 = st.sidebar.number_input("自動止損(%)", min_value=0.1, max_value=50.0, step=0.1, key="ui_sl_pct")

_write_json_file(
    UI_PREFS_PATH,
    {
        "symbol": str(st.session_state.get("ui_symbol", 預設交易對)),
        "kline_interval": str(st.session_state.get("ui_interval", "1h")),
        "kline_count": int(st.session_state.get("ui_kline_count", 300)),
        "backtest_sample_rows": int(st.session_state.get("ui_backtest_sample_rows", 0)),
        "kline_auto_update_enabled": bool(st.session_state.get("ui_kline_auto_update_enabled", False)),
        "kline_auto_update_sec": int(st.session_state.get("ui_kline_auto_update_sec", 15)),
        "max_train_rows": int(st.session_state.get("ui_max_train_rows", 40000)),
        "risk_profile": str(st.session_state.get("risk_profile", "中立 ⚖️")),
        "auto_trade_enabled": bool(st.session_state.get("ui_auto_trade_enabled", False)),
        "auto_trade_sec": int(st.session_state.get("ui_auto_trade_sec", 30)),
        "tp_pct": float(st.session_state.get("ui_tp_pct", 1.5)),
        "sl_pct": float(st.session_state.get("ui_sl_pct", 1.0)),
        "black_swan_reserve_usdt": float(st.session_state.get("ui_black_swan_reserve_usdt", 0.0)),
        "black_swan_threshold": float(st.session_state.get("ui_black_swan_threshold", 1.0)),
        "updated_at_utc": _utc_now_iso(),
    },
)

def _run_okx(action: str) -> dict | None:
    import os

    os.environ["OKX_INST_ID"] = okx_inst
    os.environ["OKX_NOTIONAL_USDT"] = str(float(okx_notional))
    os.environ["OKX_BLACK_SWAN_RESERVE_USDT"] = str(float(黑天鵝保留資金))
    os.environ["OKX_BLACK_SWAN_THRESHOLD"] = str(float(黑天鵝觸發門檻))
    os.environ["OKX_ENABLE_TRADING"] = "1" if okx_enable else "0"
    os.environ["OKX_SIMULATED"] = "1"
    os.environ["OKX_MAX_LEVERAGE"] = "100"
    with st.spinner("OKX 模擬盤下單中..."):
        try:
            if okx_sync_before_order:
                try:
                    os.environ["KLINE_KEEP_ROWS"] = str(_resolve_keep_rows_for_runtime(週期))
                    run_quick_update(symbol=交易對, interval=週期)
                except Exception as e:
                    st.sidebar.warning(f"資料快速更新失敗：{e}")
            trade_res = execute_latest_signal_okx(OUTPUT_DIR, 交易對, 週期,
                                                   leverage_override=0, action_override=action)
            _act_msg = str(trade_res.get("action", ""))
            if _act_msg == "HOLD":
                _note = str((trade_res.get("risk_controls", {}) or {}).get("note", "")).strip()
                if _note:
                    st.sidebar.info(f"OKX HOLD（{action}，不送單）：{_note}")
                else:
                    st.sidebar.info(f"OKX HOLD（{action}，不送單）")
            else:
                st.sidebar.success(f"OKX 完成：{_act_msg} ({action})")
            st.session_state["okx_last"] = trade_res
            act = str(trade_res.get("action", ""))
            if act != "HOLD":
                append_okx_order_record(
                    outputs_dir=OUTPUT_DIR,
                    source="dashboard_manual" if action in {"LONG", "SHORT", "CLOSE"} else "dashboard_auto",
                    symbol=str(交易對),
                    interval=str(週期),
                    trade_res=trade_res,
                    control_payload={
                        "mode": "pure_ai",
                        "okx_inst_id": str(okx_inst),
                        "okx_notional_usdt": float(okx_notional),
                        "okx_black_swan_reserve_usdt": float(st.session_state.get("ui_black_swan_reserve_usdt", 0.0)),
                        "okx_black_swan_threshold": float(st.session_state.get("ui_black_swan_threshold", 1.0)),
                        "okx_enable_trading": bool(okx_enable),
                        "okx_simulated": True,
                        "okx_max_leverage": 100,
                    },
                )
            px = float(trade_res.get("price", 0.0) or 0.0)
            if act == "OPEN_LONG":
                st.session_state["auto_pos_state"] = {
                    "side": "long", "entry": px,
                    "opened_at": str(trade_res.get("decision", {}).get("timestamp", ""))
                }
            elif act == "OPEN_SHORT":
                st.session_state["auto_pos_state"] = {
                    "side": "short", "entry": px,
                    "opened_at": str(trade_res.get("decision", {}).get("timestamp", ""))
                }
            elif act == "CLOSE":
                st.session_state["auto_pos_state"] = None
            return trade_res
        except Exception as e:
            st.sidebar.error(f"OKX 失敗：{e}")
            return None


# ── 按鈕動作 ────────────────────────────────────────────────────────────────
def _prepare_data_train_env(interval: str | None = None) -> int:
    os.environ["TRAIN_DEVICE"] = "cloud"
    os.environ["NPU_STRICT"] = "0"
    _user_limit = int(訓練最大樣本數)
    _backtest_limit = int(回測樣本數)
    _effective_train_rows = int(max(0, _user_limit))
    _target_interval = str(interval or 週期)
    _floor_keep_rows = int(週期資料門檻.get(_target_interval, 每週期K線顯示保底))
    _effective_keep_rows = max(
        _floor_keep_rows,
        int(_user_limit) if _user_limit > 0 else 0,
        int(_backtest_limit) if _backtest_limit > 0 else 0,
    )
    os.environ["KLINE_KEEP_ROWS"] = str(_effective_keep_rows)
    os.environ["MAX_TRAIN_ROWS"] = str(_effective_train_rows)
    return _effective_train_rows


def _resolve_keep_rows_for_runtime(interval: str | None = None) -> int:
    _user_limit = int(訓練最大樣本數)
    _backtest_limit = int(回測樣本數)
    _target_interval = str(interval or 週期)
    _floor_keep_rows = int(週期資料門檻.get(_target_interval, 每週期K線顯示保底))
    return int(
        max(
            _floor_keep_rows,
            int(_user_limit) if _user_limit > 0 else 0,
            int(_backtest_limit) if _backtest_limit > 0 else 0,
        )
    )


if 按鈕全量:
    _train_cap = _prepare_data_train_env(週期)
    with st.spinner("正在更新全部週期資料並訓練（資料保留量會依訓練/回測樣本自動調整）..."):
        _failed: list[str] = []
        for _tf in 多週期清單:
            try:
                _prepare_data_train_env(_tf)
                run_pipeline(force_full_refresh=True, symbol=交易對, interval=_tf)
            except Exception as e:
                _failed.append(f"{_tf}: {e}")
    if _failed:
        st.error("部分週期失敗：" + " | ".join(_failed))
    else:
        st.success(f"全部週期完成（{', '.join(多週期清單)}），每週期訓練筆數上限：{_train_cap}")

if 按鈕快速:
    _prepare_data_train_env(週期)
    with st.spinner("快速更新全部週期中（只更新最新資料，不重訓）..."):
        _failed: list[str] = []
        for _tf in 多週期清單:
            try:
                os.environ["KLINE_KEEP_ROWS"] = str(_resolve_keep_rows_for_runtime(_tf))
                run_quick_update(symbol=交易對, interval=_tf)
            except FileNotFoundError:
                _failed.append(f"{_tf}: 尚未有模型")
            except Exception as e:
                _failed.append(f"{_tf}: {e}")
    if _failed:
        st.warning("快速更新完成，但部分週期未成功：" + " | ".join(_failed))
    else:
        st.success(f"快速更新完成（{', '.join(多週期清單)}）")

if 按鈕增量重訓:
    _train_cap = _prepare_data_train_env(週期)
    進度文字 = st.empty()
    進度條 = st.progress(0, text="準備開始...")

    _failed: list[str] = []
    _total = len(多週期清單)
    for _idx, _tf in enumerate(多週期清單):
        _base = (_idx * 100.0) / max(1, _total)
        _span = 100.0 / max(1, _total)

        def 回報進度(p: int, msg: str, _tf_in: str = _tf, _base_in: float = _base, _span_in: float = _span) -> None:
            p = max(0, min(100, int(p)))
            overall = int(min(99, _base_in + (_span_in * p / 100.0)))
            進度條.progress(overall, text=f"[{_tf_in}] {msg} ({overall}%)")
            進度文字.info(f"目前進度：[{_tf_in}] {msg}")

        try:
            _prepare_data_train_env(_tf)
            run_pipeline(force_full_refresh=False, progress_cb=回報進度, symbol=交易對, interval=_tf)
        except Exception as e:
            _failed.append(f"{_tf}: {e}")

    進度條.progress(100, text="全部完成 (100%)")
    if _failed:
        st.error("部分週期訓練失敗：" + " | ".join(_failed))
    else:
        st.success(f"增量更新 + 重訓回測完成（全部週期），每週期訓練筆數上限：{_train_cap}")

# ── Teacher 蒸餾訓練 ──────────────────────────────────────────────────
if 按鈕訓練Teacher:
    if not _teacher_csv.exists():
        st.error(f"找不到訊號檔：{_teacher_csv.name}，請先執行「增量更新+重訓回測」產生資料。")
    else:
        from src.distillation import train_teacher as _train_teacher
        _t_進度文字 = st.empty()
        _t_進度條 = st.progress(0, text="Teacher 訓練準備中...")

        def _teacher_cb(p: int, msg: str) -> None:
            _p = max(0, min(100, int(p)))
            _t_進度條.progress(_p, text=f"{msg} ({_p}%)")
            _t_進度文字.info(f"🧑‍🏫 Teacher：{msg}")

        with st.spinner("訓練 Teacher 集成模型中（RF×500 + GradientBoosting），請稍候..."):
            try:
                _t_rpt = _train_teacher(
                    csv_path=_teacher_csv,
                    model_dir=BASE_DIR / "models",
                    output_dir=OUTPUT_DIR,
                    symbol=交易對,
                    interval=週期,
                    max_rows=int(_teacher_max_rows),
                    temperature=float(_teacher_temperature),
                    n_rf_estimators=int(_teacher_n_rf),
                    gb_n_estimators=200,
                    progress_cb=_teacher_cb,
                )
                _t_進度條.progress(100, text="Teacher 訓練完成！(100%)")
                _stats = _t_rpt.get("soft_label_stats", {})
                st.success(
                    f"✅ Teacher 訓練完成！平均信心："
                    f"{_stats.get('mean_teacher_confidence', 0)*100:.1f}%  |  "
                    f"平均槓桿：{_stats.get('mean_teacher_leverage', 0):.2f}×  |  "
                    f"槓桿 MAE：{_t_rpt.get('leverage_mae', 0):.4f}"
                )
                st.session_state["teacher_report_cache"] = _t_rpt
            except Exception as _e:
                st.error(f"Teacher 訓練失敗：{_e}")

if st.session_state.get("_sync_after_interval_switch", False) and not (按鈕全量 or 按鈕快速 or 按鈕增量重訓):
    with st.sidebar:
        with st.spinner("已切換週期，正在同步..."):
            try:
                os.environ["KLINE_KEEP_ROWS"] = str(_resolve_keep_rows_for_runtime(週期))
                run_quick_update(symbol=交易對, interval=週期)
                st.success("週期切換同步完成")
            except FileNotFoundError:
                st.info("該週期尚無模型，請先執行「增量更新+重訓回測」。")
            except Exception as e:
                st.warning(f"週期切換同步失敗：{e}")
    st.session_state["_sync_after_interval_switch"] = False

目前時間 = time.time()
if 即時更新啟用:
    上次更新 = float(st.session_state.get("kline_auto_last_update_ts", 0.0))
    if (目前時間 - 上次更新) >= int(即時更新秒數):
        try:
            os.environ["KLINE_KEEP_ROWS"] = str(_resolve_keep_rows_for_runtime(週期))
            run_quick_update(symbol=交易對, interval=週期)
            st.session_state["kline_auto_last_update_ts"] = 目前時間
            st.session_state["kline_auto_last_msg"] = f"K線即時更新成功：{time.strftime('%H:%M:%S')}"
        except FileNotFoundError:
            st.session_state["kline_auto_last_msg"] = "K線即時更新失敗：尚未有該週期模型。"
        except Exception as e:
            st.session_state["kline_auto_last_msg"] = f"K線即時更新失敗：{e}"
    if st.session_state.get("kline_auto_last_msg"):
        st.sidebar.caption(str(st.session_state["kline_auto_last_msg"]))

signals = 讀取訊號資料()
report = 讀取報告()
trades_df = 讀取交易明細(交易對, 週期)

if signals.empty:
    raw_path = BASE_DIR / "data" / f"{交易對}_{週期}_ohlcv.csv"
    raw_last = "未知"
    if raw_path.exists():
        try:
            raw_last = str(pd.read_csv(raw_path, usecols=["timestamp"])["timestamp"].iloc[-1])
        except Exception:
            raw_last = "讀取失敗"
    raw_last_utc, raw_last_tw = _format_ts_dual(raw_last)
    st.warning(
        f"目前沒有此週期的模型訊號檔：`{目前訊號檔.name}`。\n\n"
        f"原始K線檔最新開盤時間：`{raw_last_utc}`（台北：`{raw_last_tw}`）。\n\n"
        "請先按左側「增量更新+重訓回測」建立該週期訊號後再顯示模型結果。"
    )
    st.stop()

if "okx_last" in st.session_state:
    with st.expander("OKX 模擬盤下單回應"):
        st.json(st.session_state["okx_last"])

with st.expander("OKX 歷史紀錄", expanded=False):
    _okx_since_tw = pd.Timestamp("2026-05-20 00:00:00", tz="Asia/Taipei")
    _okx_since_utc = _okx_since_tw.tz_convert("UTC")
    _okx_hist = load_okx_order_history(OUTPUT_DIR, since_utc=_okx_since_utc)
    if _okx_hist.empty:
        st.caption("尚無 OKX 歷史紀錄（目前僅顯示 2026-05-20 起）。")
    else:
        st.caption("僅顯示 2026-05-20（台北）起的 OKX 下單紀錄。")
        _show_cols = [
            c
            for c in [
                "logged_at_utc",
                "source",
                "symbol",
                "interval",
                "decision_timestamp",
                "signal",
                "action",
                "price",
                "leverage",
                "size",
                "enable_trading",
                "inst_id",
            ]
            if c in _okx_hist.columns
        ]
        st.dataframe(_okx_hist.sort_values("logged_at_utc", ascending=False)[_show_cols].head(200), use_container_width=True)

最新 = signals.iloc[-1]
價格 = 取得數值(最新, "close")
P看漲 = 取得數值(最新, "p_long")
P看跌 = 取得數值(最新, "p_short")
P觀望 = 取得數值(最新, "p_flat")
模型槓桿 = 取得數值(最新, "suggested_leverage", 1.0)
安全槓桿 = 取得數值(最新, "max_safe_leverage", 1.0)
信心指數 = _direction_confidence(P看漲, P看跌, P觀望)

# ── AI 自動判斷風格 ─────────────────────────────────────────────────────────
ai_style_label, ai_style_key, ai_style_score = _ai_classify_style(最新)

# 實際執行槓桿 = AI建議 × 偏好係數 × 上限
執行槓桿 = min(模型槓桿 * 槓桿偏好係數, float(槓桿上限), float(安全槓桿))
執行槓桿 = max(1.0, round(執行槓桿, 2))

最新開盤UTC, 最新開盤台北 = _format_ts_dual(最新["timestamp"])
推測週期秒數 = _infer_interval_seconds_from_signals(signals)
最新收盤時間 = pd.NaT
if 推測週期秒數 > 0:
    最新收盤時間 = _to_utc_timestamp(最新["timestamp"]) + pd.Timedelta(seconds=推測週期秒數 - 1)
最新收盤UTC, 最新收盤台北 = _format_ts_dual(最新收盤時間)

目前設定 = Settings(symbol=交易對, interval=週期)

# 訊號判斷（門檻由 AI 信心決定，不被用戶鎖定）
_signal_threshold = 目前設定.get_signal_threshold()
訊號, 動作, 顏色類 = 判斷訊號(P看漲, P看跌, _signal_threshold)

監控視窗 = min(len(signals), 168)
if 推測週期秒數 > 0:
    每週K線數 = max(48, int(round((7 * 24 * 3600) / 推測週期秒數)))
    監控視窗 = min(len(signals), 每週K線數)
if 監控視窗 < 48:
    監控視窗 = min(len(signals), 48)
drift_alerts = compute_drift_alerts(
    signals,
    window=max(1, 監控視窗),
    confidence_floor=max(0.35, _signal_threshold - 0.03),
)
系統狀態圖示, 系統狀態文字 = get_system_status(drift_alerts)
觸發警報 = [a.message for a in drift_alerts if a.triggered]
regime_key = str(最新.get("regime", "ranging") or "ranging").lower()
市場狀態 = {
    "trend": "趨勢盤",
    "volatile": "高波動",
    "ranging": "盤整盤",
}.get(regime_key, "未判定")

# ── 自動交易邏輯 ─────────────────────────────────────────────────────────────
if 自動交易啟用:
    if not okx_enable:
        st.sidebar.warning("已開啟純AI自動交易，但目前未允許下單。請勾選『允許送出模擬盤下單』。")
    else:
        pos = st.session_state.get("auto_pos_state")
        if isinstance(pos, dict) and pos.get("entry") and pos.get("side") in {"long", "short"}:
            entry = float(pos["entry"])
            side = str(pos["side"])
            tp = float(自動止盈百分比) / 100.0
            sl = float(自動止損百分比) / 100.0
            if side == "long":
                tp_hit = 價格 >= entry * (1.0 + tp)
                sl_hit = 價格 <= entry * (1.0 - sl)
                pnl = (價格 / max(entry, 1e-9)) - 1.0
            else:
                tp_hit = 價格 <= entry * (1.0 - tp)
                sl_hit = 價格 >= entry * (1.0 + sl)
                pnl = (entry / max(價格, 1e-9)) - 1.0
            st.sidebar.caption(
                f"持倉監控：{side} 入場 {entry:,.2f}，現價 {價格:,.2f}，浮動 {pnl*100:.2f}%"
            )
            if tp_hit:
                st.sidebar.success("觸發自動止盈，執行平倉。")
                _run_okx("CLOSE")
                st.session_state["auto_pos_state"] = None
                st.session_state["auto_trade_last_msg"] = f"止盈出場 (TP {float(自動止盈百分比):.1f}%) 入場:{entry:,.2f}"
            elif sl_hit:
                st.sidebar.error("觸發自動止損，執行平倉。")
                _run_okx("CLOSE")
                st.session_state["auto_pos_state"] = None
                st.session_state["auto_trade_last_msg"] = f"止損出場 (SL {float(自動止損百分比):.1f}%) 入場:{entry:,.2f}"

        上次訊號簽名 = str(st.session_state.get("auto_trade_last_signal_sig", ""))
        目前訊號簽名 = f"{最新['timestamp']}|{int(取得數值(最新,'signal',0))}|{float(取得數值(最新,'suggested_leverage',1.0)):.2f}"
        if 目前訊號簽名 != 上次訊號簽名 and not st.session_state.get("auto_pos_state"):
            _auto_res = _run_okx("AUTO")
            st.session_state["auto_trade_last_signal_sig"] = 目前訊號簽名
            _risk = (_auto_res or {}).get("risk_controls", {}) if isinstance(_auto_res, dict) else {}
            if bool(_risk.get("hold_due_to_black_swan", False)):
                st.session_state["auto_trade_last_msg"] = "黑天鵝風險啟動：自動交易暫停，請手動操作倉位。"
            else:
                st.session_state["auto_trade_last_msg"] = f"純AI自動交易已執行（新訊號）：{最新['timestamp']}"
        if st.session_state.get("auto_trade_last_msg"):
            st.sidebar.caption(str(st.session_state["auto_trade_last_msg"]))


# ═══════════════════ MAIN UI ════════════════════════════════════════════════

# ── AI 風格卡片 ──────────────────────────────────────────────────────────────
style_class_map = {"激進 🔥": "style-aggressive", "中立 ⚖️": "style-neutral", "保守 🛡️": "style-conservative"}
style_cn_map = {"激進 🔥": "激進", "中立 ⚖️": "中立", "保守 🛡️": "保守"}
style_desc_map = {
    "激進 🔥": f"高貪婪指數 / 低波動 / 強趨勢 → AI 採積極策略（評分 {ai_style_score:+.2f}）",
    "中立 ⚖️": f"市場訊號均衡 → AI 採穩健策略（評分 {ai_style_score:+.2f}）",
    "保守 🛡️": f"高恐懼 / 高波動 / 低信心 → AI 採保守策略（評分 {ai_style_score:+.2f}）",
}

col_style, col_conf = st.columns([3, 2])
with col_style:
    st.markdown(
        f"""<div class="style-card">
            <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.1em;">
              🤖 AI 自動判斷風格
            </div>
            <span class="style-badge {style_class_map[ai_style_key]}">
              {ai_style_label}
            </span>
            &nbsp;
            <span style="color:#94a3b8;font-size:0.9rem;">
              用戶偏好：{style_cn_map[風險偏好]}（槓桿係數 {槓桿偏好係數:.0%}）
            </span>
            <div style="color:#64748b;font-size:0.82rem;margin-top:6px;">
              {style_desc_map[ai_style_key]}
            </div>
          </div>""",
        unsafe_allow_html=True,
    )
with col_conf:
    conf_pct = min(100, int(信心指數 * 100))
    conf_color = "#22c55e" if conf_pct >= 50 else ("#facc15" if conf_pct >= 25 else "#ef4444")
    st.markdown(
        f"""<div class="style-card" style="height:100%;display:flex;flex-direction:column;justify-content:center;">
            <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.1em;">
              🎯 AI 信心指數
            </div>
            <div style="font-size:2.5rem;font-weight:800;color:{conf_color};line-height:1.2;">
              {conf_pct}%
            </div>
            <div class="conf-bar-bg" style="margin-top:6px;">
              <div class="conf-bar-fill" style="width:{conf_pct}%;background:{conf_color};"></div>
            </div>
          </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ── 頂部指標 ─────────────────────────────────────────────────────────────────
餘額顯示 = "未同步"
if _bal_data:
    if _bal_data.get("usdt") is not None:
        餘額顯示 = f"{float(_bal_data['usdt']):,.2f} USDT"
    elif _bal_data.get("totalEq"):
        餘額顯示 = f"{_bal_data['totalEq']} USDT"
    elif _bal_data.get("error"):
        餘額顯示 = "讀取失敗"

頂部 = st.columns([1.45, 1.35, 1.10, 1.10, 1.00])
metrics = [
    ("OKX 餘額", 餘額顯示, ""),
    ("最新收盤價", f"{價格:,.2f} USDT", ""),
    ("看漲機率", f"{P看漲 * 100:.2f}%", "signal-bull"),
    ("看跌機率", f"{P看跌 * 100:.2f}%", "signal-bear"),
    ("執行槓桿", f"{執行槓桿:.2f}×", ""),
]
for col, (title, val, cls) in zip(頂部, metrics):
    with col:
        color_style = f'color:{"#22c55e" if cls=="signal-bull" else "#ef4444" if cls=="signal-bear" else "#f0f4ff"};'
        st.markdown(
            f"""<div class="metric-card">
                  <div class="metric-title">{title}</div>
                  <div class="metric-value" style="{color_style}">{val}</div>
                </div>""",
            unsafe_allow_html=True,
        )

監控列 = st.columns([1.0, 1.6])
with 監控列[0]:
    st.markdown(
        f"""<div class="metric-card">
              <div class="metric-title">市場狀態</div>
              <div class="metric-value">{市場狀態}</div>
              <div class="subtle">目前週期：{週期} | 訊號門檻：{_signal_threshold:.2f}</div>
            </div>""",
        unsafe_allow_html=True,
    )
with 監控列[1]:
    _alert_text = " | ".join(觸發警報[:2]) if 觸發警報 else f"最近 {監控視窗} 根 K 線內未發現明顯漂移。"
    st.markdown(
        f"""<div class="metric-card">
              <div class="metric-title">模型監控</div>
              <div class="metric-value" style="font-size:1.45rem;">{系統狀態圖示} {系統狀態文字}</div>
              <div class="subtle">{_alert_text}</div>
            </div>""",
        unsafe_allow_html=True,
    )

控制列 = st.columns([1.35, 1.15])
with 控制列[0]:
    st.markdown("""<div class="metric-card"><div class="metric-title">黑天鵝緩衝金 (USDT)</div></div>""", unsafe_allow_html=True)
    st.number_input(
        "黑天鵝緩衝金 (USDT)",
        min_value=0.0,
        max_value=10_000_000.0,
        step=10.0,
        key="ui_black_swan_reserve_usdt",
        label_visibility="collapsed",
        help="黑天鵝事件觸發時，純AI自動交易會暫停；此金額視為你保留手動操作的資金。",
    )
with 控制列[1]:
    st.markdown("""<div class="metric-card"><div class="metric-title">黑天鵝觸發門檻（風險分數）</div></div>""", unsafe_allow_html=True)
    st.number_input(
        "黑天鵝觸發門檻（風險分數）",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        key="ui_black_swan_threshold",
        label_visibility="collapsed",
        help="無單位分數（0~10）。當 black_swan_risk_score 超過此值時，自動交易暫停並改為手動操作。",
    )

st.markdown(
    f'<div class="signal-line" style="margin:12px 0 4px">訊號: <span class="{顏色類}">{訊號}</span>'
    f' &nbsp;|&nbsp; 動作: <span class="{顏色類}">{動作}</span></div>',
    unsafe_allow_html=True,
)

看漲原因, 看跌原因 = _build_bull_bear_reasons(最新)
原因左, 原因右 = st.columns(2)
with 原因左:
    st.markdown("#### 🟢 看漲原因")
    for text in 看漲原因:
        st.markdown(f"- {text}")
with 原因右:
    st.markdown("#### 🔴 看跌原因")
    for text in 看跌原因:
        st.markdown(f"- {text}")

st.markdown(
    f'<div class="subtle">K線開盤：{最新開盤台北} | K線收盤：{最新收盤台北} | '
    f'模型槓桿：{模型槓桿:.2f}× | 安全槓桿：{安全槓桿:.2f}× | 上限：{槓桿上限}×</div>',
    unsafe_allow_html=True,
)
_backend_note = ""
try:
    _model_dir_check = BASE_DIR / "models" / f"{交易對}_{週期}"
    if (_model_dir_check / "torch_models.pt").exists():
        try:
            import torch as _t_check  # noqa: F401
            _backend_note = "⚡ torch 加速"
        except Exception:
            _backend_note = "⚠️ sklearn 降級（torch 不可用，快速更新仍可運作）"
    elif (_model_dir_check / "signal_clf.joblib").exists():
        _backend_note = "🧠 sklearn CPU"
except Exception:
    pass
st.markdown(
    f'<div class="subtle">資料來源: Binance API | AI: SNR + MACD + 恐懼貪婪 | 模擬盤: OKX'
    + (f' | 模型後端: {_backend_note}' if _backend_note else '') + '</div>',
    unsafe_allow_html=True,
)

顯示區 = signals.tail(int(K線根數)).copy()
回測樣本區 = signals.copy()
if int(回測樣本數) > 0 and len(回測樣本區) > int(回測樣本數):
    回測樣本區 = 回測樣本區.tail(int(回測樣本數)).reset_index(drop=True)
目前回測曲線, 目前回測報告 = run_backtest(回測樣本區.copy(), 目前設定, interval=週期) if not 回測樣本區.empty else (pd.DataFrame(), {})
回測起始 = _format_ts_dual(回測樣本區["timestamp"].iloc[0]) if not 回測樣本區.empty else ("N/A", "N/A")
回測結束 = _format_ts_dual(回測樣本區["timestamp"].iloc[-1]) if not 回測樣本區.empty else ("N/A", "N/A")
回測天數 = 0.0
if not 回測樣本區.empty:
    _bt_span = _to_utc_timestamp(回測樣本區["timestamp"].iloc[-1]) - _to_utc_timestamp(回測樣本區["timestamp"].iloc[0])
    if pd.notna(_bt_span):
        回測天數 = max(0.0, float(_bt_span.total_seconds()) / 86400.0)

# ── K線圖 ─────────────────────────────────────────────────────────────────────
fig_k = K線圖(顯示區)
if 顯示SNR:
    all_levels = []
    atr_proxy = float(顯示區["atr_14"].dropna().tail(100).median()) if "atr_14" in 顯示區.columns else (價格 * 0.002)
    merge_tol = max(atr_proxy * 0.6, 價格 * 0.0008)
    for tf in SNR候選週期:
        src = 顯示區 if tf == 週期 else _嘗試載入週期資料(交易對, tf)
        if src is None or src.empty:
            continue
        needed = [c for c in ["timestamp", "open", "high", "low", "close", "volume",
                               "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote"]
                  if c in src.columns]
        src2 = src[needed].copy()
        lv = compute_snr_levels(src2, timeframe=tf, lookback_bars=800, pivot_window=5, max_levels=int(SNR最大線數))
        all_levels.extend(lv)
    merged = merge_multitimeframe_levels(all_levels, tolerance_abs=float(merge_tol))
    _x_end = 顯示區["timestamp"].iloc[-1]
    for lv in merged:
        overlap_count = len(lv.timeframes)
        if overlap_count < int(SNR重疊層數):
            continue
        tfs = ",".join(sorted(lv.timeframes,
                               key=lambda x: ["5m", "15m", "30m", "1h", "1d"].index(x)
                               if x in ["5m", "15m", "30m", "1h", "1d"] else 99))
        kind = "S" if (lv.kinds == {"S"}) else ("R" if (lv.kinds == {"R"}) else "S/R")
        color, line_opacity, label_bg, line_width = _snr_style(overlap_count, kind)
        _touch = 顯示區[(顯示區["low"] <= float(lv.price)) & (顯示區["high"] >= float(lv.price))]
        if _touch.empty:
            continue
        _x_start = _touch["timestamp"].iloc[0]
        fig_k.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=_x_start,
            x1=_x_end,
            y0=float(lv.price),
            y1=float(lv.price),
            line=dict(color=color, width=line_width, dash="solid"),
            opacity=line_opacity,
        )
        fig_k.add_annotation(
            x=_x_end,
            y=float(lv.price),
            xref="x",
            yref="y",
            text=f"{kind} x{overlap_count} {tfs}",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(color=color, size=11),
            bgcolor=label_bg,
        )

st.plotly_chart(
    fig_k,
    use_container_width=True,
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToAdd": ["pan2d", "zoom2d", "resetScale2d"],
    },
)
_plotly_interact_config = {
    "scrollZoom": True,
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["pan2d", "zoom2d", "resetScale2d"],
}

# ── 分頁 ─────────────────────────────────────────────────────────────────────
分頁 = st.tabs([
    "📊 買賣機率",
    "📈 趨勢高低點",
    "〽️ MACD",
    "📉 ATR",
    "🗞️ 事件",
    "😨 恐懼貪婪",
    "📋 交易紀錄",
    "🏆 回測摘要",
    "🧑‍🏫 Teacher蒸餾",
], key="main_dashboard_tabs")

with 分頁[0]:
    st.plotly_chart(買賣橫條圖(P看漲, P看跌, P觀望), use_container_width=True, config=_plotly_interact_config)

with 分頁[1]:
    st.plotly_chart(趨勢高低點圖(顯示區), use_container_width=True, config=_plotly_interact_config)

with 分頁[2]:
    st.plotly_chart(MACD圖(顯示區), use_container_width=True, config=_plotly_interact_config)

with 分頁[3]:
    st.plotly_chart(ATR圖(顯示區), use_container_width=True, config=_plotly_interact_config)

with 分頁[4]:
    st.markdown("### 🗞️ 事件清單（左：過去 / 右：未來）")
    目前風險分數 = 取得數值(最新, "market_panic_score", 0.0)
    st.caption(
        f"目前事件風險分數：`{目前風險分數:.2f}`（由戰爭/恐慌新聞、黑天鵝訊號、CPI/PPI/FOMC 時段等特徵組合）"
    )
    st.caption("新聞源每小時更新，會特別追蹤『美聯儲 / 川普 / 戰爭 / 恐慌』關鍵字。")
    if 取得數值(最新, "war_news_score", 0.0) > 0:
        st.error("偵測到戰爭/地緣衝突新聞，請優先檢查倉位風險。")
    elif 取得數值(最新, "panic_news_score", 0.0) > 0:
        st.warning("偵測到市場恐慌訊號，策略會偏向防守或反向避險。")

    左欄, 右欄 = st.columns(2)
    with 左欄:
        st.markdown("#### 過去事件")
        過去事件表 = _build_past_event_table(signals, max_rows=50)
        if 過去事件表.empty:
            st.info("最近沒有顯著事件。")
        else:
            st.dataframe(過去事件表, use_container_width=True, hide_index=True)
    with 右欄:
        st.markdown("#### 未來事件")
        未來事件表 = _build_future_event_table(pd.Timestamp.utcnow(), days=120)
        if 未來事件表.empty:
            st.info("目前沒有未來事件資料。")
        else:
            st.dataframe(未來事件表, use_container_width=True, hide_index=True)

with 分頁[5]:
    st.plotly_chart(恐懼貪婪儀表(取得數值(最新, "fear_greed_value", 50.0)), use_container_width=True, config=_plotly_interact_config)

# ─── 交易紀錄分頁（新） ────────────────────────────────────────────────────────
with 分頁[6]:
    st.markdown("### 📋 AI 交易紀錄")

    if trades_df.empty:
        st.info("尚未產生交易明細。請先執行一次「增量更新+重訓回測」或「快速更新」。")
    else:
        # 累計盈虧折線圖（僅顯示台北時區「今日 00:00」之後）
        _chart_df = trades_df.copy()
        _x_col = "出場時間" if "出場時間" in _chart_df.columns else _chart_df.columns[0]
        _x_ts = pd.to_datetime(_chart_df[_x_col], utc=True, errors="coerce")
        _today_tw = pd.Timestamp.now(tz="Asia/Taipei").normalize()
        _today_utc = _today_tw.tz_convert("UTC")
        _chart_df = _chart_df[_x_ts >= _today_utc].copy()
        if _chart_df.empty:
            st.info(f"今日（{_today_tw.strftime('%Y-%m-%d')}）尚無可顯示的交易圖表資料。")
        else:
            st.plotly_chart(
                盈虧折線圖(_chart_df),
                use_container_width=True,
                config=_plotly_interact_config,
            )

        st.divider()
        st.markdown("#### 逐筆交易明細")

        # 格式化表格
        display_df = 格式化交易明細(trades_df, signals)

        if display_df.empty:
            _raw = trades_df.copy()
            for _c in _raw.select_dtypes(include="object").columns:
                _raw[_c] = _raw[_c].astype(str)
            st.dataframe(_safe_df(_raw), use_container_width=True, hide_index=True)
        else:
            # 排序控制
            c1, c2 = st.columns([3, 2])
            with c1:
                sort_col = st.selectbox("排序欄位", options=list(display_df.columns), index=0,
                                         key="trade_sort_col_new")
            with c2:
                sort_desc = st.checkbox("降冪排序", value=True, key="trade_sort_desc_new")

            sorted_df = display_df.sort_values(
                by=sort_col, ascending=not sort_desc, na_position="last"
            ).reset_index(drop=True)

            # 顏色標記：方向欄位
            def _style_row(row: pd.Series):
                styles = [""] * len(row)
                if "方向" in display_df.columns:
                    idx = list(display_df.columns).index("方向")
                    if idx < len(styles):
                        if str(row.iloc[idx]) == "多":
                            styles[idx] = "color:#22c55e;font-weight:700"
                        elif str(row.iloc[idx]) == "空":
                            styles[idx] = "color:#ef4444;font-weight:700"
                if "盈虧(USDT)" in display_df.columns:
                    idx2 = list(display_df.columns).index("盈虧(USDT)")
                    if idx2 < len(styles):
                        try:
                            val = float(str(row.iloc[idx2]).replace("%", ""))
                            styles[idx2] = "color:#22c55e" if val >= 0 else "color:#ef4444"
                        except Exception:
                            pass
                return styles

            # 確保所有字串欄位是純 str，數値欄位是純 float，避免 Arrow 序列化錯誤
            _display_df = sorted_df.copy()
            for _c in _display_df.columns:
                if _c in ["盈虧%", "看漲機率", "看跌機率", "信心指數", "AI風格", "方向", "進場時間", "出場時間"]:
                    _display_df[_c] = _display_df[_c].astype(str)
                elif _c in ["進場價", "出場價", "槓桿", "盈虧(USDT)"]:
                    _display_df[_c] = pd.to_numeric(_display_df[_c], errors="coerce")
                elif _c == "持倉K數":
                    _display_df[_c] = pd.to_numeric(_display_df[_c], errors="coerce").astype("Int64")

            st.dataframe(
                _display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "盈虧(USDT)": st.column_config.NumberColumn("盈虧(USDT)", format="%.3f"),
                    "進場價": st.column_config.NumberColumn("進場價", format="%.2f"),
                    "出場價": st.column_config.NumberColumn("出場價", format="%.2f"),
                    "槓桿": st.column_config.NumberColumn("槓桿", format="%.2f×"),
                } if hasattr(st, "column_config") else None,
            )

            # 統計摘要
            st.divider()
            st.markdown("#### 📊 快速統計")
            stat_cols = st.columns(4)
            total_trades = len(sorted_df)
            direction_col = "方向" if "方向" in sorted_df.columns else None
            long_cnt = int((sorted_df["方向"] == "多").sum()) if direction_col else 0
            short_cnt = int((sorted_df["方向"] == "空").sum()) if direction_col else 0

            with stat_cols[0]:
                st.metric("總交易筆數", total_trades)
            with stat_cols[1]:
                st.metric("做多次數 🟢", long_cnt)
            with stat_cols[2]:
                st.metric("做空次數 🔴", short_cnt)
            with stat_cols[3]:
                if "盈虧(USDT)" in sorted_df.columns:
                    try:
                        total_pnl = sorted_df["盈虧(USDT)"].astype(float).sum()
                        st.metric("總盈虧(USDT)", f"{total_pnl:+.3f}",
                                   delta_color="normal" if total_pnl >= 0 else "inverse")
                    except Exception:
                        st.metric("總盈虧(USDT)", "N/A")
                else:
                    st.metric("總盈虧(USDT)", "N/A")

with 分頁[7]:
    bt = 目前回測報告 if isinstance(目前回測報告, dict) else {}
    _wf_report = _read_json_file(OUTPUT_DIR / f"walkforward_report_{tag}.json")
    _wf_is_stale = _is_walkforward_stale(_wf_report, len(回測樣本區), str(回測樣本區["timestamp"].iloc[-1]) if not 回測樣本區.empty else "")
    _wf_for_warning = None if _wf_is_stale else _wf_report
    _bt_severity, _bt_warnings = _build_backtest_warnings(週期, len(回測樣本區), 回測天數, bt, _wf_for_warning)
    if not bt:
        st.info("尚未找到回測報告。")
    else:
        if _bt_severity == "critical":
            st.error("回測可信度警告：" + " | ".join(_bt_warnings))
        elif _bt_severity == "warning":
            st.warning("回測可信度提醒：" + " | ".join(_bt_warnings))
        else:
            st.success(_bt_warnings[0])
        st.caption("勝率 / 獲利因子 / 盈虧比 / 交易筆數使用逐筆交易口徑；Sharpe / Sortino / Calmar / VaR / ES 使用權益曲線逐 K 口徑。")
        st.markdown(
            f'<div class="subtle">回測樣本：{len(回測樣本區):,} 根 K 線 | 起點：{回測起始[1]} | 終點：{回測結束[1]} | 期間：約 {回測天數:,.1f} 天</div>',
            unsafe_allow_html=True,
        )
        _tm = report.get("train_metrics", {}) if isinstance(report.get("train_metrics"), dict) else {}
        _split_cols = st.columns(3)
        _split_cards = [
            ("訓練樣本", f"{int(_tm.get('train_rows', 0) or 0):,} 根", "模型實際訓練使用的樣本數"),
            ("驗證樣本", f"{int(_tm.get('test_rows', 0) or 0):,} 根", "模型訓練時保留的樣本外驗證區"),
            ("回測樣本", f"{len(回測樣本區):,} 根", "目前回測摘要使用的樣本數"),
        ]
        for _col, (_title, _value, _hint) in zip(_split_cols, _split_cards):
            with _col:
                st.markdown(
                    f"""<div class="metric-card">
                          <div class="metric-title">{_title}</div>
                          <div class="metric-value">{_value}</div>
                          <div class="subtle">{_hint}</div>
                        </div>""",
                    unsafe_allow_html=True,
                )
        trade_items = [
            ("回測K線數", 回測顯示值("回測K線數", bt.get("rows"))),
            ("交易筆數", 回測顯示值("交易筆數", bt.get("trades"))),
            ("勝率", 回測顯示值("勝率", bt.get("win_rate"))),
            ("獲利因子", 回測顯示值("獲利因子", bt.get("profit_factor"))),
            ("盈虧比", 回測顯示值("盈虧比", bt.get("pnl_ratio"))),
            ("平均槓桿", 回測顯示值("平均槓桿", bt.get("avg_leverage"))),
            ("最大使用槓桿", 回測顯示值("最大使用槓桿", bt.get("max_leverage_used"))),
        ]
        curve_items = [
            ("總收益", 回測顯示值("總收益", bt.get("total_return"))),
            ("最大回撤", 回測顯示值("最大回撤", bt.get("max_drawdown"))),
            ("Sharpe", 回測顯示值("Sharpe", bt.get("sharpe"))),
            ("Sortino", 回測顯示值("Sortino", bt.get("sortino"))),
            ("Calmar", 回測顯示值("Calmar", bt.get("calmar"))),
            ("VaR 95%", 回測顯示值("VaR 95%", bt.get("var_95"))),
            ("ES 95%", 回測顯示值("ES 95%", bt.get("es_95"))),
        ]
        c_top1, c_top2 = st.columns(2)
        with c_top1:
            st.markdown("#### 交易級指標")
            trade_df = pd.DataFrame(trade_items, columns=["指標", "數值"])
            st.dataframe(_safe_df(trade_df), use_container_width=True, hide_index=True)
        with c_top2:
            st.markdown("#### 權益曲線指標")
            curve_df = pd.DataFrame(curve_items, columns=["指標", "數值"])
            st.dataframe(_safe_df(curve_df), use_container_width=True, hide_index=True)

        st.markdown("#### Walk-Forward")
        _wf_cols = st.columns([1.2, 2.8])
        with _wf_cols[0]:
            if st.button("重跑 Walk-Forward", use_container_width=True, key=f"wf_run_{tag}"):
                try:
                    with st.spinner("正在重跑 walk-forward 驗證..."):
                        _wf_report = run_walkforward_validation(回測樣本區.copy(), 目前設定, n_folds=4)
                        save_walkforward_report(_wf_report, OUTPUT_DIR, tag)
                    st.success("Walk-forward 驗證已更新。")
                except Exception as e:
                    st.error(f"Walk-forward 失敗：{e}")
        with _wf_cols[1]:
            if _wf_is_stale:
                st.info("目前的 walk-forward 報告是舊樣本版本，請按左側按鈕重跑後再判讀樣本外結果。")
            elif _wf_report:
                _wf_sum = _wf_report.get("summary", {})
                st.markdown(
                    f'<div class="subtle">來源樣本：{int(_wf_report.get("source_rows", 0)):,} 根 | fold 數：{int(_wf_report.get("fold_count", 0))} | 每 fold 測試：約 {int(_wf_report.get("test_rows_per_fold", 0)):,} 根</div>',
                    unsafe_allow_html=True,
                )
                wf_items = [
                    ("樣本外總收益", 回測顯示值("總收益", _wf_sum.get("compounded_total_return"))),
                    ("平均 fold 收益", 回測顯示值("總收益", _wf_sum.get("average_fold_return"))),
                    ("中位數 fold 收益", 回測顯示值("總收益", _wf_sum.get("median_fold_return"))),
                    ("平均 fold 勝率", 回測顯示值("勝率", _wf_sum.get("average_fold_win_rate"))),
                    ("平均 fold 盈虧比", 回測顯示值("盈虧比", _wf_sum.get("average_fold_pnl_ratio"))),
                    ("平均 fold 期望值", 回測顯示值("期望值", _wf_sum.get("average_fold_expectancy_unit"))),
                    ("平均 fold Sharpe", 回測顯示值("Sharpe", _wf_sum.get("average_fold_sharpe"))),
                    ("最差 fold 回撤", 回測顯示值("最大回撤", -abs(float(_wf_sum.get("worst_fold_drawdown", 0.0) or 0.0)))),
                    ("總樣本外交易筆數", int(_wf_sum.get("total_fold_trades", 0) or 0)),
                    ("正期望 fold 數", f"{int(_wf_sum.get('positive_expectancy_folds', 0) or 0)}/{int(_wf_report.get('fold_count', 0) or 0)}"),
                ]
                st.dataframe(_safe_df(pd.DataFrame(wf_items, columns=["指標", "數值"])), use_container_width=True, hide_index=True)
                _fold_rows = []
                for _fold in _wf_report.get("folds", []):
                    _b = _fold.get("backtest_report", {}) if isinstance(_fold.get("backtest_report"), dict) else {}
                    _fold_rows.append(
                        {
                            "Fold": int(_fold.get("fold", 0) or 0),
                            "TrainRows": int(_fold.get("train_rows", 0) or 0),
                            "TestRows": int(_fold.get("test_rows", 0) or 0),
                            "TestStart": _format_tw(_fold.get("test_start_utc")),
                            "TestEnd": _format_tw(_fold.get("test_end_utc")),
                            "TotalReturn": 回測顯示值("總收益", _b.get("total_return")),
                            "WinRate": 回測顯示值("勝率", _b.get("win_rate")),
                            "MaxDD": 回測顯示值("最大回撤", _b.get("max_drawdown")),
                            "Sharpe": 回測顯示值("Sharpe", _b.get("sharpe")),
                            "Trades": int(_b.get("trades", 0) or 0),
                        }
                    )
                if _fold_rows:
                    st.dataframe(_safe_df(pd.DataFrame(_fold_rows)), use_container_width=True, hide_index=True)
            else:
                st.info("尚未產生 walk-forward 報告。點左側按鈕即可開始。")

# ── Teacher 蒸餾分頁 (index 8) ───────────────────────────────────────────────
import plotly.graph_objects as _pgo

with 分頁[8]:
    st.markdown("### 🧑‍🏫 Teacher 模型蒸餾報告")
    from src.distillation import (
        teacher_exists as _tch_exists,
        load_teacher_report as _load_tch_rpt,
        load_teacher_soft_labels as _load_soft,
    )

    if not _tch_exists(BASE_DIR / "models", 交易對, 週期):
        st.info("尚未訓練 Teacher 模型。請在左側「🧑‍🏫 知識蒸餾 (Teacher)」區域點擊「訓練 Teacher 模型」。")
        st.markdown("""
| 角色 | 說明 |
|------|------|
| **Teacher** | RF×500 + GradientBoosting 大型集成，訓練慢但品質高 |
| **軟標籤** | Teacher 輸出的機率分佈（溫度縮放），比 0/1 更豐富 |
| **Student** | 現有小模型，未來學習 Teacher 軟標籤提升準確率 |
| **溫度 T** | T=2 → 機率更平滑，讓 Student 更容易學習不確定性 |
        """)
    else:
        _rpt = st.session_state.get("teacher_report_cache") or _load_tch_rpt(OUTPUT_DIR, 交易對, 週期)
        if _rpt:
            _m = _rpt.get("meta", {})
            _stats = _rpt.get("soft_label_stats", {})
            _cls_rpt = _rpt.get("classification_report", {})
            _tc1, _tc2, _tc3, _tc4 = st.columns(4)
            _tc1.metric("📊 訓練筆數", f"{_m.get('n_rows', 0):,}")
            _tc2.metric("🌡️ 溫度 T", f"{_m.get('temperature', 2.0):.1f}")
            _tc3.metric("🧠 平均信心", f"{_stats.get('mean_teacher_confidence', 0)*100:.1f}%")
            _tc4.metric("💰 平均槓桿", f"{_stats.get('mean_teacher_leverage', 0):.2f}×")
            _tc5, _tc6, _tc7, _tc8 = st.columns(4)
            _tc5.metric("RF 樹數", f"{_m.get('n_rf_estimators', 0)} 棵")
            _tc6.metric("GB 迭代數", f"{_m.get('gb_n_estimators', 0)}")
            _tc7.metric("槓桿 MAE", f"{_rpt.get('leverage_mae', 0):.4f}")
            _wf1 = _cls_rpt.get("weighted avg", {}).get("f1-score", 0)
            _tc8.metric("F1 (weighted)", f"{_wf1:.4f}")
            st.divider()
            _sl = _load_soft(OUTPUT_DIR, 交易對, 週期)
            if not _sl.empty:
                _x = (pd.to_datetime(_sl["timestamp"], utc=True, errors="coerce") if "timestamp" in _sl.columns else list(range(len(_sl))))
                _fig_t = _pgo.Figure()
                for _col, _color, _name in [("soft_p_long", "#22c55e", "🟢 看漲"), ("soft_p_short", "#ef4444", "🔴 看跌"), ("soft_p_flat", "#64748b", "⚪ 觀望")]:
                    if _col in _sl.columns:
                        _fig_t.add_trace(_pgo.Scatter(x=_x, y=_sl[_col], name=_name, line=dict(color=_color, width=1.2), mode="lines"))
                _fig_t.update_layout(template="plotly_dark", height=300, margin=dict(l=20,r=20,t=36,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", dragmode="pan", title=dict(text=f"Teacher 軟標籤機率（T={_m.get('temperature',2.0)}）", font=dict(size=13,color="#94a3b8")), legend=dict(orientation="h",y=-0.2))
                st.plotly_chart(_fig_t, use_container_width=True, config=_plotly_interact_config)
                _fig_t2 = _pgo.Figure()
                if "teacher_confidence" in _sl.columns:
                    _fig_t2.add_trace(_pgo.Scatter(x=_x, y=(_sl["teacher_confidence"]*100).round(2), name="Teacher 信心%", line=dict(color="#a78bfa",width=1.5), mode="lines"))
                if "teacher_leverage" in _sl.columns:
                    _fig_t2.add_trace(_pgo.Scatter(x=_x, y=_sl["teacher_leverage"].round(2), name="Teacher 槓桿", line=dict(color="#f59e0b",width=1.5), mode="lines", yaxis="y2"))
                _fig_t2.update_layout(template="plotly_dark", height=260, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", dragmode="pan", yaxis=dict(title="信心 %",side="left"), yaxis2=dict(title="槓桿",overlaying="y",side="right"), legend=dict(orientation="h",y=-0.25))
                st.plotly_chart(_fig_t2, use_container_width=True, config=_plotly_interact_config)
                st.markdown("**軟標籤樣本（最新 50 筆）**")
                _sl_show = _sl.tail(50).copy()
                for _nc in ["soft_p_long","soft_p_short","soft_p_flat","raw_p_long","raw_p_short","teacher_confidence","teacher_leverage"]:
                    if _nc in _sl_show.columns:
                        _sl_show[_nc] = pd.to_numeric(_sl_show[_nc], errors="coerce").round(4)
                if "teacher_signal" in _sl_show.columns:
                    _sl_show["teacher_signal"] = _sl_show["teacher_signal"].astype(str)
                st.dataframe(_safe_df(_sl_show), use_container_width=True, hide_index=True)
                st.caption(f"📂 軟標籤：outputs/teacher_soft_labels_{交易對}_{週期}.csv | 模型：models/teacher/{交易對}_{週期}/")
            else:
                st.info("軟標籤 CSV 尚未產生，請先點擊「訓練 Teacher」。")
        else:
            st.warning("找不到 Teacher 報告檔案。")

# ── 自動刷新循環（非阻塞，避免 sleep+rerun 導致前端殘留重複 UI） ───────────────────
if 即時更新啟用 or 自動交易啟用:
    refresh_s = int(min(
        int(即時更新秒數) if 即時更新啟用 else 3600,
        int(自動交易秒數) if 自動交易啟用 else 3600
    ))
    st.sidebar.caption(f"⏱ 自動循環中，每 {refresh_s} 秒刷新一次。")
    _reload_ms = max(1000, int(refresh_s) * 1000)
    components.html(
        f"""
        <script>
        (function() {{
          const ms = {_reload_ms};
          if (window.__aiDashboardReloadTimer) {{
            clearTimeout(window.__aiDashboardReloadTimer);
          }}
          window.__aiDashboardReloadTimer = setTimeout(function() {{
            window.location.reload();
          }}, ms);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )
