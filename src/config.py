from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Settings:
    symbol: str | None = None
    interval: str | None = None
    market_type: str | None = None
    base_url: str | None = None
    kline_limit: int | None = None

    data_dir: Path | None = None
    output_dir: Path | None = None
    model_dir: Path | None = None

    hours_lookback_overlap: int | None = None
    min_train_rows: int | None = None

    future_horizon_hours: int | None = None
    long_threshold: float | None = None
    short_threshold: float | None = None

    fee_bps: float | None = None
    slippage_bps: float | None = None
    max_leverage: int | None = None
    drawdown_stop: float | None = None

    cryptopanic_auth_token: str | None = None
    quick_window_days: int | None = None
    train_device: str | None = None
    npu_strict: bool | None = None
    torch_epochs: int | None = None
    torch_batch_size: int | None = None
    max_train_rows: int | None = None
    promote_min_win_rate_delta: float | None = None
    promote_min_total_return_delta: float | None = None
    promote_max_drawdown_increase: float | None = None
    promote_min_profit_factor_delta: float | None = None
    promote_min_trades: int | None = None

    funding_rate_8h_bps: float | None = None
    interval_signal_thresholds: dict | None = None

    def __post_init__(self) -> None:
        self.symbol = self.symbol or os.getenv("SYMBOL", "BTCUSDT")
        self.interval = self.interval or os.getenv("INTERVAL", "1h")
        self.market_type = self.market_type or os.getenv("MARKET_TYPE", "spot")
        self.base_url = self.base_url or os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
        self.kline_limit = int(self.kline_limit or os.getenv("KLINE_LIMIT", "1000"))

        self.data_dir = self.data_dir or Path(os.getenv("DATA_DIR", "data"))
        self.output_dir = self.output_dir or Path(os.getenv("OUTPUT_DIR", "outputs"))
        self.model_dir = self.model_dir or Path(os.getenv("MODEL_DIR", "models"))

        self.hours_lookback_overlap = int(self.hours_lookback_overlap or os.getenv("HOURS_LOOKBACK_OVERLAP", "240"))
        self.min_train_rows = int(self.min_train_rows or os.getenv("MIN_TRAIN_ROWS", "500"))

        self.future_horizon_hours = int(self.future_horizon_hours or os.getenv("FUTURE_HORIZON_HOURS", "12"))
        self.long_threshold = float(self.long_threshold or os.getenv("LONG_THRESHOLD", "0.005"))
        self.short_threshold = float(self.short_threshold or os.getenv("SHORT_THRESHOLD", "-0.005"))

        self.fee_bps = float(self.fee_bps or os.getenv("FEE_BPS", "6"))
        self.slippage_bps = float(self.slippage_bps or os.getenv("SLIPPAGE_BPS", "4"))
        self.max_leverage = int(self.max_leverage or os.getenv("MAX_LEVERAGE", "100"))
        self.drawdown_stop = float(self.drawdown_stop or os.getenv("DRAWDOWN_STOP", "0.35"))

        self.cryptopanic_auth_token = self.cryptopanic_auth_token or os.getenv("CRYPTOPANIC_AUTH_TOKEN", "")
        self.quick_window_days = int(self.quick_window_days or os.getenv("QUICK_WINDOW_DAYS", "7"))
        self.train_device = str(self.train_device or os.getenv("TRAIN_DEVICE", "cloud")).lower()
        self.npu_strict = bool(int(self.npu_strict if self.npu_strict is not None else os.getenv("NPU_STRICT", "0")))
        self.torch_epochs = int(self.torch_epochs or os.getenv("TORCH_EPOCHS", "5"))
        self.torch_batch_size = int(self.torch_batch_size or os.getenv("TORCH_BATCH_SIZE", "4096"))
        # 0 means use all rows. Set a positive number to speed up retraining.
        self.max_train_rows = int(self.max_train_rows if self.max_train_rows is not None else os.getenv("MAX_TRAIN_ROWS", "40000"))
        self.promote_min_win_rate_delta = float(
            self.promote_min_win_rate_delta
            if self.promote_min_win_rate_delta is not None
            else os.getenv("MODEL_PROMOTE_MIN_WIN_RATE_DELTA", "0.01")
        )
        self.promote_min_total_return_delta = float(
            self.promote_min_total_return_delta
            if self.promote_min_total_return_delta is not None
            else os.getenv("MODEL_PROMOTE_MIN_TOTAL_RETURN_DELTA", "0.0")
        )
        self.promote_max_drawdown_increase = float(
            self.promote_max_drawdown_increase
            if self.promote_max_drawdown_increase is not None
            else os.getenv("MODEL_PROMOTE_MAX_DRAWDOWN_INCREASE", "0.02")
        )
        self.promote_min_profit_factor_delta = float(
            self.promote_min_profit_factor_delta
            if self.promote_min_profit_factor_delta is not None
            else os.getenv("MODEL_PROMOTE_MIN_PROFIT_FACTOR_DELTA", "0.0")
        )
        self.promote_min_trades = int(
            self.promote_min_trades
            if self.promote_min_trades is not None
            else os.getenv("MODEL_PROMOTE_MIN_TRADES", "20")
        )

        self.funding_rate_8h_bps = float(self.funding_rate_8h_bps if self.funding_rate_8h_bps is not None else os.getenv('FUNDING_RATE_8H_BPS', '2.5'))
        self.interval_signal_thresholds = self.interval_signal_thresholds or {
            '5m': 0.60, '15m': 0.55, '30m': 0.52,
            '1h': 0.48, '1d': 0.42,
        }

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def get_signal_threshold(self) -> float:
        """Return per-interval signal threshold."""
        return float((self.interval_signal_thresholds or {}).get(self.interval or '1h', 0.48))
