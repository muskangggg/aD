"""
compute_credit_score.py

Builds Credit Score CS = repayment rate + LTV trajectory proxy + account
age + liquidation history, from the real Aave wallet data pulled in
fetch_wallet_history.py.

Usage:
    python compute_credit_score.py

Notes:
- Run this AFTER re-running fetch_wallet_history.py with the USD-value
  fix applied (uses assetPriceUSD x decimals-normalized amount), or
  repayment_rate/ltv will be computed on stale/mixed-unit totals.
- LTV here is a summary ratio across a wallet's full history
  (total borrowed USD / total supplied USD), not a point-in-time
  trajectory -- a true trajectory would need LTV computed at each
  transaction timestamp, which is a possible refinement later if you
  want time-series CS features instead of a single summary score.
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_wallet_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop wallets with no real activity at all -- nothing to score.
    df = df[df["total_transactions"] > 0].reset_index(drop=True)
    return df


def normalize_column(s: pd.Series, clip_quantile: float = 0.95) -> pd.Series:
    """Min-max scale to [0, 1], clipping extreme outliers first (a few
    wallets with huge tx counts would otherwise squash everyone else to
    near-zero)."""
    upper = s.quantile(clip_quantile)
    s_clipped = s.clip(upper=upper)
    s_min, s_max = s_clipped.min(), s_clipped.max()
    if s_max == s_min:
        return s_clipped * 0
    return (s_clipped - s_min) / (s_max - s_min)


def compute_credit_score(df: pd.DataFrame, weights: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)) -> pd.DataFrame:
    """CS = weighted combo of repayment rate, borrow/supply ratio (inverted --
    lower ratio is "safer"), account age, and liquidation history (inverted --
    fewer liquidations is "safer"). Output in [0, 1], higher = better credit."""
    w_repay, w_ltv, w_age, w_liq = weights

    df = df.copy()
    df["repayment_rate_filled"] = df["repayment_rate"].fillna(0)  # never borrowed = no repayment history to reward
    df["repayment_norm"] = normalize_column(df["repayment_rate_filled"])

    df["ltv_filled"] = df["ltv"].fillna(0)  # never borrowed = no LTV risk to penalize
    df["ltv_norm"] = 1 - normalize_column(df["ltv_filled"])  # invert: lower LTV -> higher score

    df["age_norm"] = normalize_column(df["account_age_days"].fillna(0))

    df["liq_norm"] = 1 - normalize_column(df["num_liquidations"])  # invert: fewer liquidations -> higher score

    df["credit_score"] = (
        w_repay * df["repayment_norm"]
        + w_ltv * df["ltv_norm"]
        + w_age * df["age_norm"]
        + w_liq * df["liq_norm"]
    )

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wallet_path = os.path.join(RAW_DIR, "wallet_credit_features.csv")
    df = load_wallet_data(wallet_path)
    print(f"Loaded {len(df)} active wallets")

    df = compute_credit_score(df)

    out_path = os.path.join(OUTPUT_DIR, "wallet_credit_scores.csv")
    cols = [
        "wallet", "repayment_rate", "ltv", "account_age_days",
        "num_liquidations", "credit_score",
    ]
    df[cols].to_csv(out_path, index=False)

    print(f"  -> saved to {out_path}")
    print(f"  -> credit_score range: {df['credit_score'].min():.3f} to {df['credit_score'].max():.3f}")
    print(f"  -> mean credit_score: {df['credit_score'].mean():.3f}")
    print(f"\nWallets with liquidation history: {(df['num_liquidations'] > 0).sum()} / {len(df)}")


if __name__ == "__main__":
    main()