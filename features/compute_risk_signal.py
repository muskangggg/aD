"""
compute_risk_signal.py  (v3 -- forward-looking Risk Signal: liquidation probability)

What changed in v3 (6 Nov, upgrade plan Step 2):
  The ablation (30 Oct) showed that agents WITHOUT R were just as proactive
  as agents with R. Reason: R v2 was mostly health_deficit, and
  health_deficit is only HF rescaled -- and HF is already in the
  observation. Like telling someone the temperature in Celsius AND in
  Fahrenheit: the second number adds nothing.

  R v3 answers a question HF alone cannot answer:
      "What is the chance this loan gets liquidated in the next K days?"
  It combines the SAME ingredients as before (how healthy the loan is, how
  jumpy the market is, how the position is built) through a probability
  model instead of hand-picked weights:

      R = P(HF falls below 1.0 at any time in the next K days)
        = 2 * Phi( -d / (sigma * sqrt(K)) )          (capped at 1)

      d     = log distance between today's ETH price and the price at
              which HF would hit 1.0 ("distance to liquidation")
      sigma = today's daily volatility estimate (EWMA, RiskMetrics 1996)
      2*Phi = reflection principle for a driftless random walk: the chance
              of TOUCHING the barrier at any time is about twice the chance
              of ENDING below it.
  Same idea as Merton's (1974) "distance to default" in credit risk.

  Example (ETH collateral, USDC debt, K = 5 days):
      HF 1.3, calm market   (daily vol 3.5%) -> R about 0.1%
      HF 1.3, stormy market (daily vol 7%)   -> R about 9.5%
  Same HF, about 100x different risk. HF says how far the cliff is;
  R also says how fast you are driving.

  The old functions (health deficit, utilization, rolling volatility) are
  kept unchanged so old code and old results still work.

  compute_risk_signal(...) keeps its old 3-argument call so the training,
  evaluation and diagnosis scripts did not need changes: in v3 its first
  argument is the liquidation probability and R = that probability.

Usage:
    python features/compute_risk_signal.py      # prints a table of R values
"""

import math
import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

VOL_WINDOW_DAYS = 30  # rolling window for realized volatility (old, data file only)

WARNING_HF = 1.5       # (v2) health deficit starts rising below this HF
LIQUIDATION_HF = 1.0   # (v2) health deficit reaches 1.0 here

# ---- v3: forward-looking risk ----
RISK_HORIZON_DAYS = 5     # K: "chance of liquidation in the next K days"
EWMA_LAMBDA = 0.94        # RiskMetrics (J.P. Morgan, 1996) daily decay factor


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
    v_min, v_max = vol_series.min(), vol_series.max()
    if v_max == v_min:
        return vol_series * 0
    return (vol_series - v_min) / (v_max - v_min)


# ---- Functions used by the Gym environment (and the live agent service) ----

def compute_health_deficit(health_factor: float,
                           warning_hf: float = WARNING_HF,
                           liquidation_hf: float = LIQUIDATION_HF) -> float:
    """(v2, kept for logs) 0 when HF >= warning_hf, rises linearly to 1 at liquidation_hf."""
    if health_factor >= warning_hf:
        return 0.0
    return float(np.clip((warning_hf - health_factor) / (warning_hf - liquidation_hf), 0.0, 1.0))


def compute_utilization(total_debt_usd: float, total_collateral_usd: float) -> float:
    """Fraction of collateral value currently borrowed against."""
    if total_collateral_usd <= 0:
        return 0.0
    return max(0.0, min(1.0, total_debt_usd / total_collateral_usd))


def update_ewma_variance(prev_var: float, log_return: float, lam: float = EWMA_LAMBDA) -> float:
    """RiskMetrics EWMA: today's variance = lam * yesterday's + (1 - lam) * today's return^2.
    Reacts within a few days when the market turns stormy."""
    return lam * prev_var + (1.0 - lam) * log_return ** 2


def liquidation_distance(coll_eth_usd: float, coll_usdc: float, debt_eth_usd: float, debt_usdc: float,
                         liquidation_threshold: float = 0.8):
    """How far (in log price) the ETH price must move until HF = 1.0.

    HF < 1  <=>  a * x < b,  where x = (future ETH price / today's price),
        a = LT * collateral_ETH - debt_ETH      (ETH-linked cushion)
        b = debt_USDC - LT * collateral_USDC    (USDC-linked gap)
    Returns (d, already_liquidatable):
        d = |ln(x*)| (None if no ETH price move can liquidate the loan)
    Works for every mix: ETH collateral + USDC debt (price must FALL),
    USDC collateral + ETH debt (price must RISE), or both."""
    lt = liquidation_threshold
    if debt_eth_usd + debt_usdc <= 1e-9:
        return None, False
    a = lt * coll_eth_usd - debt_eth_usd
    b = debt_usdc - lt * coll_usdc
    already = a * 1.0 < b                    # HF < 1 at today's price
    if already:
        return 0.0, True
    if abs(a) < 1e-12:                       # HF does not depend on the ETH price
        return None, False
    x_star = b / a
    if x_star <= 0:                          # no positive price can liquidate it
        return None, False
    return abs(math.log(x_star)), False


def compute_liquidation_probability(d, already_liquidatable: bool, sigma_daily: float,
                                    horizon_days: int = RISK_HORIZON_DAYS) -> float:
    """P(price touches the liquidation price within horizon_days), driftless random walk."""
    if already_liquidatable:
        return 1.0
    if d is None:
        return 0.0
    spread = max(sigma_daily, 1e-9) * math.sqrt(horizon_days)
    z = d / spread
    p = 2.0 * 0.5 * math.erfc(z / math.sqrt(2.0))   # 2 * Phi(-z)
    return float(min(1.0, max(0.0, p)))


def compute_risk_signal(liquidation_probability: float,
                        volatility_norm: float = None,
                        utilization: float = None) -> float:
    """v3: R = liquidation probability over the next RISK_HORIZON_DAYS.
    The last two arguments are accepted only so older scripts keep working
    (volatility and the position are already INSIDE the probability)."""
    return float(liquidation_probability)


def main():
    print(f"Risk Signal v3: P(liquidation within {RISK_HORIZON_DAYS} days), "
          f"ETH collateral + USDC debt\n")
    vols = (0.02, 0.035, 0.07)
    print("   HF  " + "  ".join(f"vol {v:.1%}/day" for v in vols))
    for hf in (2.0, 1.5, 1.3, 1.2, 1.15, 1.1, 1.05, 1.0):
        # collateral 1000 ETH-USD, debt chosen so HF = hf
        d, already = liquidation_distance(1000.0, 0.0, 0.0, 800.0 / hf)
        cells = [f"{compute_liquidation_probability(d, already, v):>14.2%}" for v in vols]
        print(f"  {hf:.2f}" + "".join(cells))
    print("\n-> same HF, very different risk when the market is calm vs stormy.")

    d, already = liquidation_distance(0.0, 1000.0, 800.0 / 1.3, 0.0)   # USDC collateral, ETH debt
    print(f"Check (USDC collateral, ETH debt, HF 1.3, vol 3.5%): d = {d:.3f} "
          f"(price must RISE {math.exp(d) - 1:.0%}), R ={compute_liquidation_probability(d, already, 0.035):.2%}")


if __name__ == "__main__":
    main()