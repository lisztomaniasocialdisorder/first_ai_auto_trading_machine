from __future__ import annotations

import json
import importlib.util
import time
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

from src.pipeline import run_pipeline, run_quick_update
from src.snr import compute_snr_levels, merge_multitimeframe_levels
from src.paper_trade_okx import execute_latest_signal_okx

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

    confidence = max(p_long, p_short) - p_flat

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


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
預設交易對 = "BTCUSDT"

st.set_page_config(page_title="BTC AI 智能交易儀表板", layout="wide", page_icon="🤖")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
      * { font-family: 'Inter', sans-serif !important; }
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
    </style>
    """,
    unsafe_allow_html=True,
)


def 讀取訊號資料() -> pd.DataFrame:
    if not 目前訊號檔.exists():
        舊檔 = OUTPUT_DIR / "signals_with_features.csv"
        if 週期 == "1h" and 舊檔.exists():
            df = pd.read_csv(舊檔)
        else:
            return pd.DataFrame()
    else:
        df = pd.read_csv(目前訊號檔)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
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
    return pd.read_csv(p)


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
    if 指標 in {"勝率", "最大回撤"}:
        return f"{v * 100:.2f}%"
    return v


def 判斷訊號(p_long: float, p_short: float, 門檻: float) -> tuple[str, str, str]:
    if p_long >= 門檻 and p_long > p_short:
        return "看漲", "買入 / 做多", "signal-bull"
    if p_short >= 門檻 and p_short > p_long:
        return "看跌", "賣出 / 做空", "signal-bear"
    return "觀望", "等待", "signal-flat"


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
            fig.add_trace(
                go.Scatter(
                    x=buy_df["timestamp"],
                    y=buy_df["low"] * 0.998,
                    mode="markers+text",
                    name="買點",
                    marker=dict(symbol="triangle-up", size=13, color="#22c55e",
                                line=dict(color="#166534", width=1)),
                    text=["▲"] * len(buy_df),
                    textposition="top center",
                    textfont=dict(color="#22c55e", size=10),
                )
            )
        if not sell_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_df["timestamp"],
                    y=sell_df["high"] * 1.002,
                    mode="markers+text",
                    name="賣點",
                    marker=dict(symbol="triangle-down", size=13, color="#ef4444",
                                line=dict(color="#7f1d1d", width=1)),
                    text=["▼"] * len(sell_df),
                    textposition="bottom center",
                    textfont=dict(color="#ef4444", size=10),
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
    )
    fig.update_yaxes(
        title="價格 (USDT)",
        range=[y_low - y_pad, y_high + y_pad],
        autorange=False,
        gridcolor="#1e293b",
    )
    fig.update_xaxes(gridcolor="#1e293b")
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
        uirevision="macd-static",
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
    )
    return fig


def ATR圖(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["atr_14"], name="ATR(14)",
                              line=dict(color="#a78bfa", width=2)))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["atr_pct"] * 100, name="ATR%",
                              line=dict(color="#f43f5e", width=2)))
    fig.update_layout(
        template="plotly_dark", height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
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
    if not items:
        items.append("目前無明顯事件訊號")
    return items


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
                lambda t: f"{max(0.0, _match_num(t,'p_long') + _match_num(t,'p_short') - _match_num(t,'p_flat',0.34)) * 100:.1f}%"
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
交易對 = st.sidebar.text_input("交易對", value=預設交易對)
週期 = st.sidebar.selectbox("K線週期", ["5m", "15m", "30m", "1h", "1d"], index=3)
if "_last_interval_selected" not in st.session_state:
    st.session_state["_last_interval_selected"] = 週期
if st.session_state["_last_interval_selected"] != 週期:
    st.session_state["_last_interval_selected"] = 週期
    st.session_state["_sync_after_interval_switch"] = True
槓桿上限 = 同步槓桿設定("槓桿上限", "槓桿上限 (1~100)", default=100)
K線根數 = st.sidebar.slider("K線顯示根數", min_value=100, max_value=2000, value=300, step=50)

st.sidebar.divider()
st.sidebar.markdown("### 🎯 風險偏好")
st.sidebar.caption("AI 自動判斷市場風格（激進/中立/保守）；此選項只作為槓桿倍率的偏好係數。")
風險偏好 = st.sidebar.radio(
    "槓桿偏好",
    options=list(RISK_PROFILES.keys()),
    index=1,
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
按鈕全量 = st.sidebar.button("重新抓取 BTCUSDT 全歷史", use_container_width=True)
按鈕快速 = st.sidebar.button("快速更新", use_container_width=True)
按鈕增量重訓 = st.sidebar.button("增量更新+重訓回測", use_container_width=True)
即時更新啟用 = st.sidebar.checkbox("K線即時更新", value=False,
                                     help="開啟後會定時做『快速更新』並自動刷新頁面。")
即時更新秒數 = st.sidebar.number_input("K線即時更新秒數", min_value=5, max_value=3600, value=15, step=1)
訓練最大樣本數 = st.sidebar.number_input("重訓最大樣本數 (0=全量)", min_value=0, max_value=2000000,
                                           value=40000, step=10000)
if st.sidebar.button("環境自檢", use_container_width=True):
    spec_torch = importlib.util.find_spec("torch")
    spec_dml = importlib.util.find_spec("torch_directml")
    env_info = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "cwd": str(Path.cwd()),
        "venv311_exists": (BASE_DIR / ".venv311" / "Scripts" / "python.exe").exists(),
        "venv_exists": (BASE_DIR / ".venv" / "Scripts" / "python.exe").exists(),
        "torch_installed": bool(spec_torch),
        "torch_directml_installed": bool(spec_dml),
    }
    if spec_torch:
        try:
            import torch  # type: ignore
            env_info["torch_version"] = str(torch.__version__)
        except Exception as e:
            env_info["torch_version_error"] = str(e)
    if spec_dml:
        try:
            import torch_directml  # type: ignore
            env_info["torch_directml_device"] = str(torch_directml.device())
        except Exception as e:
            env_info["torch_directml_error"] = str(e)
    st.sidebar.json(env_info)

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
okx_notional = st.sidebar.number_input("下單名目(USDT)", min_value=5.0, max_value=100000.0,
                                         value=50.0, step=5.0)
okx_enable = st.sidebar.checkbox("允許送出模擬盤下單(OKX_ENABLE_TRADING=1)", value=False)
okx_sync_before_order = st.sidebar.checkbox("下單前先快速同步資料", value=False)
自動交易啟用 = st.sidebar.checkbox("啟用純AI自動交易", value=False,
                                     help="開啟後定時用最新AI訊號自動下單。")
自動交易秒數 = st.sidebar.number_input("自動交易檢查秒數", min_value=10, max_value=3600, value=30, step=5)
自動止盈百分比 = st.sidebar.number_input("自動止盈(%)", min_value=0.1, max_value=50.0, value=1.5, step=0.1)
自動止損百分比 = st.sidebar.number_input("自動止損(%)", min_value=0.1, max_value=50.0, value=1.0, step=0.1)


def _run_okx(action: str) -> None:
    import os
    os.environ["OKX_INST_ID"] = okx_inst
    os.environ["OKX_NOTIONAL_USDT"] = str(float(okx_notional))
    os.environ["OKX_ENABLE_TRADING"] = "1" if okx_enable else "0"
    os.environ["OKX_SIMULATED"] = "1"
    os.environ["OKX_MAX_LEVERAGE"] = "100"
    with st.spinner("OKX 模擬盤下單中..."):
        try:
            if okx_sync_before_order:
                try:
                    run_quick_update(symbol=交易對, interval=週期)
                except Exception as e:
                    st.sidebar.warning(f"資料快速更新失敗：{e}")
            trade_res = execute_latest_signal_okx(OUTPUT_DIR, 交易對, 週期,
                                                   leverage_override=0, action_override=action)
            st.sidebar.success(f"OKX 完成：{trade_res.get('action')} ({action})")
            st.session_state["okx_last"] = trade_res
            act = str(trade_res.get("action", ""))
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
        except Exception as e:
            st.sidebar.error(f"OKX 失敗：{e}")


# ── 按鈕動作 ────────────────────────────────────────────────────────────────
if 按鈕全量:
    import os
    os.environ["TRAIN_DEVICE"] = "cloud"
    os.environ["NPU_STRICT"] = "0"
    os.environ["MAX_TRAIN_ROWS"] = str(int(訓練最大樣本數))
    with st.spinner("全歷史重抓 + 訓練回測中，請稍候..."):
        try:
            run_pipeline(force_full_refresh=True, symbol=交易對, interval=週期)
            st.success("全歷史更新完成")
        except Exception as e:
            st.error(f"訓練失敗：{e}")

if 按鈕快速:
    with st.spinner("快速更新中（只做尾段增量回補，不重訓）..."):
        快速成功 = True
        try:
            run_quick_update(symbol=交易對, interval=週期)
        except FileNotFoundError:
            快速成功 = False
            st.error("找不到該週期已訓練模型，請先執行一次『增量更新+重訓回測』或『全歷史重抓』。")
    if 快速成功:
        st.success("快速更新完成")

if 按鈕增量重訓:
    import os
    os.environ["TRAIN_DEVICE"] = "cloud"
    os.environ["NPU_STRICT"] = "0"
    os.environ["MAX_TRAIN_ROWS"] = str(int(訓練最大樣本數))
    進度文字 = st.empty()
    進度條 = st.progress(0, text="準備開始...")

    def 回報進度(p: int, msg: str) -> None:
        p = max(0, min(100, int(p)))
        進度條.progress(p, text=f"{msg} ({p}%)")
        進度文字.info(f"目前進度：{msg}")

    try:
        run_pipeline(force_full_refresh=False, progress_cb=回報進度, symbol=交易對, interval=週期)
        進度條.progress(100, text="全部完成 (100%)")
        st.success("增量更新 + 重訓回測完成")
    except Exception as e:
        st.error(f"訓練失敗：{e}")

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

最新 = signals.iloc[-1]
價格 = 取得數值(最新, "close")
P看漲 = 取得數值(最新, "p_long")
P看跌 = 取得數值(最新, "p_short")
P觀望 = 取得數值(最新, "p_flat")
模型槓桿 = 取得數值(最新, "suggested_leverage", 1.0)
安全槓桿 = 取得數值(最新, "max_safe_leverage", 1.0)
信心指數 = max(0.0, P看漲 + P看跌 - P觀望)

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

# 訊號判斷（門檻由 AI 信心決定，不被用戶鎖定）
_signal_threshold = 0.45  # AI 基礎門檻（固定，不讓用戶改）
訊號, 動作, 顏色類 = 判斷訊號(P看漲, P看跌, _signal_threshold)

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
            elif sl_hit:
                st.sidebar.error("觸發自動止損，執行平倉。")
                _run_okx("CLOSE")

        上次訊號簽名 = str(st.session_state.get("auto_trade_last_signal_sig", ""))
        目前訊號簽名 = f"{最新['timestamp']}|{int(取得數值(最新,'signal',0))}|{float(取得數值(最新,'suggested_leverage',1.0)):.2f}"
        if 目前訊號簽名 != 上次訊號簽名 and not st.session_state.get("auto_pos_state"):
            _run_okx("AUTO")
            st.session_state["auto_trade_last_signal_sig"] = 目前訊號簽名
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
頂部 = st.columns(4)
metrics = [
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

st.markdown(
    f'<div class="signal-line" style="margin:12px 0 4px">訊號: <span class="{顏色類}">{訊號}</span>'
    f' &nbsp;|&nbsp; 動作: <span class="{顏色類}">{動作}</span></div>',
    unsafe_allow_html=True,
)
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
    for lv in merged:
        if len(lv.timeframes) < int(SNR重疊層數):
            continue
        tfs = ",".join(sorted(lv.timeframes,
                               key=lambda x: ["5m", "15m", "30m", "1h", "1d"].index(x)
                               if x in ["5m", "15m", "30m", "1h", "1d"] else 99))
        kind = "S" if (lv.kinds == {"S"}) else ("R" if (lv.kinds == {"R"}) else "S/R")
        color = "#22c55e" if kind == "S" else ("#ef4444" if kind == "R" else "#a78bfa")
        fig_k.add_hline(
            y=float(lv.price), line_dash="solid", line_width=1,
            line_color=color, opacity=0.75,
            annotation_text=f"{kind} {tfs}",
            annotation_position="top right",
            annotation_font_color=color,
        )

st.plotly_chart(fig_k, use_container_width=True, config={"scrollZoom": True})

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
])

with 分頁[0]:
    st.plotly_chart(買賣橫條圖(P看漲, P看跌, P觀望), use_container_width=True)

with 分頁[1]:
    st.plotly_chart(趨勢高低點圖(顯示區), use_container_width=True)

with 分頁[2]:
    st.plotly_chart(MACD圖(顯示區), use_container_width=True)

with 分頁[3]:
    st.plotly_chart(ATR圖(顯示區), use_container_width=True)

with 分頁[4]:
    st.markdown("### 🗞️ 事件清單")
    for e in 事件清單(最新):
        st.write(f"• {e}")

with 分頁[5]:
    st.plotly_chart(恐懼貪婪儀表(取得數值(最新, "fear_greed_value", 50.0)), use_container_width=True)

# ─── 交易紀錄分頁（新） ────────────────────────────────────────────────────────
with 分頁[6]:
    st.markdown("### 📋 AI 交易紀錄")

    if trades_df.empty:
        st.info("尚未產生交易明細。請先執行一次「增量更新+重訓回測」或「快速更新」。")
    else:
        # 累計盈虧折線圖
        st.plotly_chart(盈虧折線圖(trades_df), use_container_width=True,
                         config={"scrollZoom": True})

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
    bt = report.get("backtest_report", {}) if report else {}
    if not bt:
        st.info("尚未找到回測報告。")
    else:
        items = [
            ("總收益", 回測顯示值("總收益", bt.get("total_return"))),
            ("最大回撤", 回測顯示值("最大回撤", bt.get("max_drawdown"))),
            ("勝率", 回測顯示值("勝率", bt.get("win_rate"))),
            ("獲利因子", 回測顯示值("獲利因子", bt.get("profit_factor"))),
            ("盈虧比", 回測顯示值("盈虧比", bt.get("pnl_ratio"))),
            ("Sharpe", 回測顯示值("Sharpe", bt.get("sharpe"))),
            ("Sortino", 回測顯示值("Sortino", bt.get("sortino"))),
            ("Calmar", 回測顯示值("Calmar", bt.get("calmar"))),
            ("VaR 95%", 回測顯示值("VaR 95%", bt.get("var_95"))),
            ("ES 95%", 回測顯示值("ES 95%", bt.get("es_95"))),
            ("平均槓桿", 回測顯示值("平均槓桿", bt.get("avg_leverage"))),
            ("最大使用槓桿", 回測顯示值("最大使用槓桿", bt.get("max_leverage_used"))),
        ]
        sum_df = pd.DataFrame(items, columns=["指標", "數值"])
        c_s1, c_s2 = st.columns([3, 2])
        with c_s1:
            summary_sort_col = st.selectbox("排序依據", options=["指標", "數值"], index=1)
        with c_s2:
            summary_desc = st.checkbox("降冪排序", value=True, key="summary_sort_desc")
        if summary_sort_col == "數值":
            sort_key = pd.to_numeric(
                sum_df["數值"].astype(str).str.replace("%", "", regex=False), errors="coerce"
            )
            sum_df = (
                sum_df.assign(_k=sort_key)
                .sort_values("_k", ascending=not summary_desc, na_position="last")
                .drop(columns="_k").reset_index(drop=True)
            )
        else:
            sum_df = sum_df.sort_values("指標", ascending=not summary_desc).reset_index(drop=True)
        _sum_safe = sum_df.copy()
        for _c in _sum_safe.select_dtypes(include="object").columns:
            _sum_safe[_c] = _sum_safe[_c].astype(str)
        st.dataframe(_sum_safe, use_container_width=True, hide_index=True)

# ── 自動刷新循環 ─────────────────────────────────────────────────────────────
if 即時更新啟用 or 自動交易啟用:
    refresh_s = int(min(
        int(即時更新秒數) if 即時更新啟用 else 3600,
        int(自動交易秒數) if 自動交易啟用 else 3600
    ))
    st.sidebar.caption(f"⏱ 自動循環中，每 {refresh_s} 秒刷新一次。")
    time.sleep(max(1, refresh_s))
    if _should_run_quick_update_now(signals, 週期):
        try:
            run_quick_update(symbol=交易對, interval=週期)
            st.session_state["kline_auto_last_msg"] = f"快速更新成功：{time.strftime('%H:%M:%S')}"
        except Exception as e:
            st.session_state["kline_auto_last_msg"] = f"快速更新失敗：{e}"
    else:
        st.session_state["kline_auto_last_msg"] = "未到新K線時間，略過快速更新。"
    st.rerun()

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
                _fig_t.update_layout(template="plotly_dark", height=300, margin=dict(l=20,r=20,t=36,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title=dict(text=f"Teacher 軟標籤機率（T={_m.get('temperature',2.0)}）", font=dict(size=13,color="#94a3b8")), legend=dict(orientation="h",y=-0.2))
                st.plotly_chart(_fig_t, use_container_width=True)
                _fig_t2 = _pgo.Figure()
                if "teacher_confidence" in _sl.columns:
                    _fig_t2.add_trace(_pgo.Scatter(x=_x, y=(_sl["teacher_confidence"]*100).round(2), name="Teacher 信心%", line=dict(color="#a78bfa",width=1.5), mode="lines"))
                if "teacher_leverage" in _sl.columns:
                    _fig_t2.add_trace(_pgo.Scatter(x=_x, y=_sl["teacher_leverage"].round(2), name="Teacher 槓桿", line=dict(color="#f59e0b",width=1.5), mode="lines", yaxis="y2"))
                _fig_t2.update_layout(template="plotly_dark", height=260, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(title="信心 %",side="left"), yaxis2=dict(title="槓桿",overlaying="y",side="right"), legend=dict(orientation="h",y=-0.25))
                st.plotly_chart(_fig_t2, use_container_width=True)
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
