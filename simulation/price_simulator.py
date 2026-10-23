"""
price_simulator.py  (v2 -- Step 2: train/test split)

Builds the "Simulated / Backtest Env" price source: each episode gets a
price path that's either (a) a real historical window sampled from actual
ETH price data, or (b) a synthetic path generated via Geometric Brownian
Motion (GBM) calibrated to that real data's drift and volatility.

What changed in v2 -- TRAIN / TEST SPLIT:
  The price history is split by time:
    - "train": the first TRAIN_FRAC (80%) of days
    - "test" : the last 20% of days, never seen during training
  Real windows are only sampled from the chosen split, and GBM is ALWAYS
  calibrated on the train part only, so no information from the test
  period leaks into training. Splitting by time (not randomly) matters for
  price data: a random split would let training windows overlap the test
  days.

Mixing: episodes mix real and synthetic paths with probability
MIX_REAL_PROB (default 70% real / 30% GBM).

Usage:
    python simulation/price_simulator.py   # demo of both splits

As a module:
    from price_simulator import PriceSimulator
    sim = PriceSimulator("../data/raw/eth_daily.csv", split="train")
    path = sim.get_episode_path(episode_length=90)
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MIX_REAL_PROB = 0.7          # probability an episode uses a real historical window vs. GBM
DEFAULT_EPISODE_LENGTH = 90  # days per episode
TRAIN_FRAC = 0.8             # first 80% of days = train, last 20% = test


class PriceSimulator:
    def __init__(self, price_csv_path: str, seed: int | None = None,
                 split: str = "train", train_frac: float = TRAIN_FRAC):
        if split not in ("train", "test", "all"):
            raise ValueError(f"split must be 'train', 'test' or 'all', got {split!r}")

        df = pd.read_csv(price_csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        all_prices = df["close"].values
        split_idx = int(len(all_prices) * train_frac)

        self.split = split
        self.train_prices = all_prices[:split_idx]
        if split == "train":
            self.prices = all_prices[:split_idx]
            self.dates = df["date"].values[:split_idx]
        elif split == "test":
            self.prices = all_prices[split_idx:]
            self.dates = df["date"].values[split_idx:]
        else:  # "all" -- old behaviour, kept only for comparison
            self.prices = all_prices
            self.dates = df["date"].values

        self.rng = np.random.default_rng(seed)

        # GBM is always calibrated on TRAIN data only (no test leakage)
        self.mu_daily, self.sigma_daily = self._calibrate_gbm(self.train_prices)

    @staticmethod
    def _calibrate_gbm(prices: np.ndarray) -> tuple[float, float]:
        """Estimate daily drift (mu) and volatility (sigma) from historical
        log returns -- these become the GBM parameters."""
        log_returns = np.diff(np.log(prices))
        return log_returns.mean(), log_returns.std()

    def _simulate_gbm_path(self, n_steps: int, s0: float | None = None) -> np.ndarray:
        """Generate one synthetic price path via GBM."""
        if s0 is None:
            s0 = float(self.rng.choice(self.prices))  # start from a plausible price level of this split

        # GBM: S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)
        shocks = self.rng.normal(
            loc=(self.mu_daily - 0.5 * self.sigma_daily ** 2),
            scale=self.sigma_daily,
            size=n_steps,
        )
        path = s0 * np.exp(np.cumsum(shocks))
        return np.concatenate([[s0], path])  # include starting price

    def _sample_real_window(self, n_steps: int) -> np.ndarray:
        """Pick a random contiguous window of real prices from this split."""
        max_start = len(self.prices) - (n_steps + 1)
        if max_start <= 0:
            # not enough history in this split for this length -- fall back to GBM
            return self._simulate_gbm_path(n_steps)
        start = self.rng.integers(0, max_start + 1)
        return self.prices[start:start + n_steps + 1]

    def get_episode_path(self, episode_length: int = DEFAULT_EPISODE_LENGTH,
                         mix_real_prob: float = MIX_REAL_PROB) -> np.ndarray:
        """Return one price path (length episode_length+1, including the
        starting price) for a single episode."""
        if self.rng.random() < mix_real_prob:
            return self._sample_real_window(episode_length)
        return self._simulate_gbm_path(episode_length)


def main():
    price_path = os.path.join(RAW_DIR, "eth_daily.csv")

    for split in ("train", "test"):
        sim = PriceSimulator(price_path, seed=42, split=split)
        print(f"=== {split.upper()} split ===")
        print(f"  days in split:   {len(sim.prices)}")
        print(f"  date range:      {str(sim.dates[0])[:10]} to {str(sim.dates[-1])[:10]}")
        print(f"  GBM (train-calibrated) annualized drift: {sim.mu_daily * 365:.4f}, "
              f"volatility: {sim.sigma_daily * np.sqrt(365):.4f}")
        print(f"  3 sample episodes (length={DEFAULT_EPISODE_LENGTH} days):")
        for i in range(3):
            path = sim.get_episode_path()
            pct = (path[-1] / path[0] - 1) * 100
            print(f"    episode {i+1}: start=${path[0]:.2f}  end=${path[-1]:.2f}  change={pct:+.1f}%")
        print()


if __name__ == "__main__":
    main()