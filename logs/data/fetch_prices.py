"""
fetch_prices.py

Pulls historical daily OHLCV price data from Binance's free public API
(no signup, no API key, ever) for the assets used in the simulation. This
is the raw price series used to:
  1. Calibrate GBM (drift, volatility) for the simulated environment
  2. Compute collateral volatility for the Risk Signal R
  3. Provide "real price data" for the backtest env

Usage:
    python fetch_prices.py

Notes:
- No API key or account needed at all -- Binance's market data endpoints
  are fully public.
- Uses /api/v3/klines, which returns candlestick (OHLCV) data. Binance
  caps each response at 1000 candles, so this script paginates
  automatically for longer histories.
- We fetch USDT-quoted pairs since that's what's actually listed on
  Binance (ETHUSDT). USDC is a stablecoin pegged near $1 -- its own
  price series has almost no variance, so if your CS/R features need
  USDC's price at all, it's effectively constant ~1.0 rather than
  something worth fetching from an exchange history.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

# ---- Config ----
BASE_URL = "https://api.binance.com/api/v3/klines"

# Binance trading pair symbols (not CoinGecko-style IDs).
# ETHUSDT = ETH priced in USDT (a dollar-pegged stablecoin, close proxy for USD).
PAIRS = {
    "ETH": "ETHUSDT",
}

DAYS_OF_HISTORY = 1000  # ~2.7 years
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw")


def fetch_klines(symbol: str, interval: str = "1d", limit: int = 1000, end_time_ms: int | None = None) -> list:
    """Fetch one page (<=1000 candles) of kline data from Binance."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if end_time_ms:
        params["endTime"] = end_time_ms

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_full_history(symbol: str, days_needed: int) -> pd.DataFrame:
    """Paginate backward in time until days_needed is covered."""
    all_rows = []
    end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    remaining = days_needed

    while remaining > 0:
        batch_size = min(remaining, 1000)
        rows = fetch_klines(symbol, limit=batch_size, end_time_ms=end_time_ms)
        if not rows:
            break

        all_rows = rows + all_rows  # prepend (older data comes first in next page)
        remaining -= len(rows)

        oldest_open_time = rows[0][0]  # kline open time in ms
        end_time_ms = oldest_open_time - 1  # step back before this page

        time.sleep(0.3)  # polite pacing, well within Binance's generous limits

        if len(rows) < batch_size:
            break  # ran out of available history

    # Kline format: [open_time, open, high, low, close, volume, close_time, ...]
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df = df[["date", "open", "high", "low", "close"]]
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, symbol in PAIRS.items():
        print(f"Fetching {DAYS_OF_HISTORY} days of {symbol} history...")
        df = fetch_full_history(symbol, DAYS_OF_HISTORY)
        out_path = os.path.join(OUTPUT_DIR, f"{name.lower()}_daily.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> saved {len(df)} rows to {out_path}")
        print(f"  -> date range: {df['date'].min()} to {df['date'].max()}")

    # USDC: stablecoin, price is ~1.0 USD by design. Not fetched from an
    # exchange -- noted here as a constant rather than wasting an API call
    # on near-zero-variance data.
    print("\nUSDC treated as a constant $1.00 peg (stablecoin) -- not fetched.")


if __name__ == "__main__":
    main()