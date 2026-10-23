"""
compute_risk_signal.py  (v2 -- early-warning health deficit)

Builds the components of Risk Signal R = health deficit + volatility + utilization.

What changed in v2 -- HEALTH DEFICIT IS NOW AN EARLY WARNING:
  Before, health_deficit was 0 for every HF >= 1.0 and only became > 0
  once HF < 1.0 -- i.e. at the moment of liquidation. That made the
  biggest part of R (weight 0.5) REACTIVE: it gave no warning at all
  before the position was already liquidatable.

  Now health_deficit starts rising as soon as HF drops below a warning
  level (WARNING_HF = 1.5) and reaches 1.0 at the liquidation level
  (HF = 1.0):
        HF >= 1.5  -> 0.0   (safe)
        HF  = 1.25 -> 0.5   (getting close)
        HF <= 1.0  -> 1.0   (liquidatable)
  This is what makes R a PROACTIVE signal (proposal Objective 2), and it
  also lets R actually reach > 0.7 near liquidation, so the PER high-risk
  boost (R > 0.7) is used.

Scope note (unchanged):
  Only the VOLATILITY component can be computed from static historical
  data. Health deficit and utilization depend on a live position, so they
  are computed inside the Gym environment at every step using the
  functions below. The right-side live agent service must use the SAME
  functions so the trained agent sees the same numbers.

Usage:
    python features/compute_risk_signal.py
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

VOL_WINDOW_DAYS = 30  # rolling window for realized volatility

WARNING_HF = 1.5       # health deficit starts rising below this HF
LIQUIDATION_HF = 1.0   # health deficit reaches 1.0 here


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


# ---- Functions used by the Gym environment (and the live agent service) ----

def compute_health_deficit(health_factor: float,
                           warning_hf: float = WARNING_HF,
                           liquidation_hf: float = LIQUIDATION_HF) -> float:
    """Early-warning distance to liquidation.
    0 when HF >= warning_hf (safe), rises linearly to 1 at liquidation_hf."""
    if health_factor >= warning_hf:
        return 0.0
    return float(np.clip((warning_hf - health_factor) / (warning_hf - liquidation_hf), 0.0, 1.0))


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
    Default weights favor health deficit as the dominant risk driver."""
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

    print("\nHealth deficit (early warning) at different HF values:")
    for hf in (2.0, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9):
        print(f"  HF={hf:.1f} -> health_deficit={compute_health_deficit(hf):.2f}")


if __name__ == "__main__":
    main()