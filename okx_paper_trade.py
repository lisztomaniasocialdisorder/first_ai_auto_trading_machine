from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.paper_trade_okx import execute_latest_signal_okx


def main() -> None:
    root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=root / ".env")
    parser = argparse.ArgumentParser(description="Execute latest model signal on OKX simulated trading (paper).")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h", help="5m/15m/30m/1h/1d")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--leverage", type=int, default=None, help="Override leverage (1..100)")
    parser.add_argument("--action", default="AUTO", choices=["AUTO", "LONG", "SHORT", "CLOSE"], help="AUTO follows signal; LONG/SHORT/CLOSE are manual")
    args = parser.parse_args()

    try:
        res = execute_latest_signal_okx(Path(args.outputs), args.symbol, args.interval, leverage_override=args.leverage, action_override=args.action)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        print("OKX 執行失敗：", msg)
        if "50101" in msg and "environment" in msg:
            print("你目前使用的 API Key 與環境不符。")
            print("如果要用模擬盤：請到 OKX『交易 -> 模擬交易(Demo Trading)』建立 Demo Trading API Key，並確保 OKX_SIMULATED=1。")
            print("如果要用正式盤：請把 OKX_SIMULATED 改成 0（並且不要再送 x-simulated-trading: 1）。")
        else:
            print("請確認專案根目錄的 .env 已填：OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
