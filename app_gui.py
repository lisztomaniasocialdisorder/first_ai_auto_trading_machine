from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

from dotenv import load_dotenv

from src.pipeline import run_pipeline


if getattr(sys, "frozen", False):
    ROOT_DIR = Path(sys.executable).resolve().parent
else:
    ROOT_DIR = Path(__file__).resolve().parent

os.chdir(ROOT_DIR)


def _resolve_report_path() -> tuple[Path, Path]:
    candidates = [
        ROOT_DIR,
        ROOT_DIR.parent,
    ]
    for base in candidates:
        rp = base / "outputs" / "report.json"
        if rp.exists():
            return base, rp
    return ROOT_DIR, ROOT_DIR / "outputs" / "report.json"


WORK_DIR, REPORT_PATH = _resolve_report_path()
WORK_DIR.joinpath("outputs").mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("BTC 1h AI Backtest")
        self.geometry("940x680")
        self.minsize(900, 620)

        self.is_running = False
        self._build_ui()
        self._load_existing_report()

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=12)
        top.pack(fill=tk.X)

        self.full_refresh_var = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(top, text="Full Refresh (refetch all history)", variable=self.full_refresh_var)
        chk.pack(side=tk.LEFT)

        self.run_btn = ttk.Button(top, text="Run: Fetch + Train + Backtest", command=self._start_pipeline)
        self.run_btn.pack(side=tk.LEFT, padx=10)

        ttk.Button(top, text="Reload report", command=self._load_existing_report).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value=f"Status: Idle | WorkDir: {WORK_DIR}")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.RIGHT)

        summary = ttk.LabelFrame(self, text="Latest Decision", padding=12)
        summary.pack(fill=tk.X, padx=12, pady=(0, 8))

        self.summary_text = tk.StringVar(value="No data yet")
        ttk.Label(summary, textvariable=self.summary_text, justify=tk.LEFT).pack(anchor="w")

        metrics = ttk.LabelFrame(self, text="Backtest/Train Summary", padding=12)
        metrics.pack(fill=tk.X, padx=12, pady=(0, 8))

        self.metrics_text = tk.StringVar(value="No data yet")
        ttk.Label(metrics, textvariable=self.metrics_text, justify=tk.LEFT).pack(anchor="w")

        log_frame = ttk.LabelFrame(self, text="Execution Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.log = ScrolledText(log_frame, height=20, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        self.is_running = running
        self.run_btn.configure(state=("disabled" if running else "normal"))
        self.status_var.set("Status: Running" if running else "Status: Idle")

    def _start_pipeline(self) -> None:
        if self.is_running:
            return
        self._set_running(True)

        self._append_log("Starting pipeline...")
        self._append_log(f"full_refresh={self.full_refresh_var.get()}")

        th = threading.Thread(target=self._run_pipeline_worker, daemon=True)
        th.start()

    def _run_pipeline_worker(self) -> None:
        try:
            load_dotenv()
            results = run_pipeline(force_full_refresh=self.full_refresh_var.get())
            self.after(0, lambda: self._on_success(results))
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            self.after(0, lambda: self._on_error(exc, tb))

    def _on_success(self, results: dict) -> None:
        self._append_log("Pipeline finished")
        self._render_results(results)
        self._set_running(False)
        messagebox.showinfo("Done", "Fetch, training, and backtest are complete.")

    def _on_error(self, exc: Exception, tb: str) -> None:
        self._append_log("Pipeline failed")
        self._append_log(str(exc))
        self._append_log(tb)
        self._set_running(False)
        messagebox.showerror("Error", f"Failed: {exc}")

    def _load_existing_report(self) -> None:
        if not REPORT_PATH.exists():
            self._append_log(f"report.json not found at: {REPORT_PATH}")
            self._append_log("Click 'Run: Fetch + Train + Backtest' to generate it.")
            return

        try:
            data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            self._render_results(data)
            self._append_log("Loaded existing report.json")
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"Failed to read report.json: {exc}")

    def _render_results(self, results: dict) -> None:
        latest = results.get("latest_decision", {})
        backtest = results.get("backtest_report", {})
        train = results.get("train_metrics", {})

        summary = (
            f"Time: {latest.get('timestamp', '-') }\n"
            f"Price: {latest.get('price', '-') }\n"
            f"Signal: {latest.get('signal', '-') } (1=buy, -1=sell, 0=hold)\n"
            f"Suggested leverage: {latest.get('suggested_leverage', '-') }\n"
            f"Max safe leverage: {latest.get('max_safe_leverage', '-') }\n"
            f"P(long): {latest.get('p_long', '-') } | P(short): {latest.get('p_short', '-') }"
        )
        self.summary_text.set(summary)

        weighted = train.get("classification_report", {}).get("weighted avg", {})
        metrics = (
            f"Total return: {backtest.get('total_return', '-') }\n"
            f"Max drawdown: {backtest.get('max_drawdown', '-') }\n"
            f"Win rate: {backtest.get('win_rate', '-') }\n"
            f"PnL ratio: {backtest.get('pnl_ratio', '-') }\n"
            f"Profit factor: {backtest.get('profit_factor', '-') }\n"
            f"Sharpe: {backtest.get('sharpe', '-') }\n"
            f"Avg leverage: {backtest.get('avg_leverage', '-') } | Max leverage used: {backtest.get('max_leverage_used', '-') }\n"
            f"F1(weighted): {weighted.get('f1-score', '-') } | leverage MAE: {train.get('leverage_mae', '-') }"
        )
        self.metrics_text.set(metrics)


if __name__ == "__main__":
    app = App()
    app.mainloop()
