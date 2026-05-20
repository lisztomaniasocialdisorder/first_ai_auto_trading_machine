from __future__ import annotations

import argparse

from dotenv import load_dotenv

from src.pipeline import pretty_print_results, run_pipeline


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="BTC 1h data pipeline: crawl, train, backtest")
    parser.add_argument("--full-refresh", action="store_true", help="Refetch full history instead of incremental update")
    args = parser.parse_args()

    results = run_pipeline(force_full_refresh=args.full_refresh)
    pretty_print_results(results)
