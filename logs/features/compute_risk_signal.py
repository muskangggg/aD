"""
compute_risk_signal.py

Builds the components of Risk Signal R = health deficit + volatility + utilization.

Important scope note:
  Only the VOLATILITY component can be computed from static historical
  data (it just needs the price series). Health deficit and utilization
  are properties of a specific borrowing position (collateral value,
  debt value, liquidation threshold) at a specific point in time -- they
  don't exist as a static number until the simulated environment (Step 5)
  is actually stepping through simulated positions.

  So this file does two things:
    1. Computes real historical price volatility from eth_daily.csv (usable now)
    2. Defines the health_deficit / utilization / composite-R formulas as
       reusable functions that the Gym environment will call at each
       simulation step, once positions exist.

Usage:
    python compute_risk_signal.py
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

VOL_WINDOW_DAYS = 30  # rolling window for realized volatility


def load_price_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def compute_rolling_volatility(df: pd.DataFrame, window: int = VOL_WINDOW_DAYS) -> pd.DataFrame:
    """Annualized realized volatility from daily log returns."""
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["log_return"].rolling(window=window).std() * np.sqrt(365)
    return df


def normalize_volatility(vol_series: pd.Series) -> pd.Series:
    """Min-max scale volatility to [0, 1] so it combines cleanly with the
    other R components (which are naturally bounded 0-1)."""
    v_min, v_max = vol_series.min(), vol_series.max()
    if v_max == v_min:
        return vol_series * 0
    return (vol_series - v_min) / (v_max - v_min)


# ---- Functions used later by the Gym environment (Step 5) ----
# These take live simulated position state, not static data, so they
# aren't run here -- they're defined here so R's formula lives in one
# place and the env just imports and calls them.

def compute_health_deficit(health_factor: float, target_hf: float = 1.0) -> float:
    """How far below a safe health factor the position currently is.
    0 = safe (HF >= target), approaches 1 as HF -> 0 (about to be liquidated)."""
    if health_factor >= target_hf:
        return 0.0
    return max(0.0, min(1.0, (target_hf - health_factor) / target_hf))


def compute_utilization(total_debt_usd: float, total_collateral_usd: float) -> float:
    """Fraction of available collateral capacity currently borrowed against."""
    if total_collateral_usd <= 0:
        return 0.0
    return max(0.0, min(1.0, total_debt_usd / total_collateral_usd))


def compute_risk_signal(
    health_deficit: float,
    volatility_norm: float,
    utilization: float,
    weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> float:
    """Composite Risk Signal R, weighted sum of the three components.
    Default weights favor health deficit as the dominant risk driver --
    tune these once you're running ablations (Step 10)."""
    w_hd, w_vol, w_util = weights
    return w_hd * health_deficit + w_vol * volatility_norm + w_util * utilization


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    price_path = os.path.join(RAW_DIR, "eth_daily.csv")
    df = load_price_data(price_path)
    df = compute_rolling_volatility(df)
    df["volatility_norm"] = normalize_volatility(df["volatility"])

    out_path = os.path.join(OUTPUT_DIR, "eth_volatility.csv")
    df[["date", "close", "log_return", "volatility", "volatility_norm"]].to_csv(out_path, index=False)

    print(f"Computed rolling {VOL_WINDOW_DAYS}-day volatility for {len(df)} days")
    print(f"  -> saved to {out_path}")
    print(f"  -> volatility range: {df['volatility'].min():.4f} to {df['volatility'].max():.4f} (annualized)")
    print("\nNote: health_deficit and utilization components of R are computed")
    print("live inside the Gym environment (Step 5) using compute_health_deficit()")
    print("and compute_utilization() from this module -- they need simulated")
    print("position state that doesn't exist as static data.")


if __name__ == "__main__":
    main()