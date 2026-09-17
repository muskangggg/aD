"""
price_simulator.py

Builds the "Simulated / Backtest Env" price source: each training episode
gets a price path that's either (a) a real historical window sampled from
actual ETH price data, or (b) a synthetic path generated via Geometric
Brownian Motion (GBM) calibrated to that same real data's drift and
volatility.

Design choice (flagged for visibility): episodes mix real and synthetic
paths with a configurable probability (default 70% real / 30% GBM). Real
windows teach the agent genuine market patterns; GBM paths fill in extra
scenario diversity -- especially tail events (sharp drops) that a single
~2.7 year window may not contain enough examples of for robust training.
Adjust MIX_REAL_PROB below if you want a different split, or set it to
1.0 / 0.0 for pure real-data or pure-GBM modes.

Usage:
    python price_simulator.py   # runs a quick self-test / demo

As a module:
    from price_simulator import PriceSimulator
    sim = PriceSimulator("../data/raw/eth_daily.csv")
    path = sim.get_episode_path(episode_length=90)
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MIX_REAL_PROB = 0.7  # probability an episode uses a real historical window vs. GBM
DEFAULT_EPISODE_LENGTH = 90  # days per training episode


class PriceSimulator:
    def __init__(self, price_csv_path: str, seed: int | None = None):
        self.df = pd.read_csv(price_csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        self.prices = self.df["close"].values
        self.rng = np.random.default_rng(seed)

        self.mu_daily, self.sigma_daily = self._calibrate_gbm()

    def _calibrate_gbm(self) -> tuple[float, float]:
        """Estimate daily drift (mu) and volatility (sigma) from real
        historical log returns -- these become the GBM parameters."""
        log_returns = np.diff(np.log(self.prices))
        mu = log_returns.mean()
        sigma = log_returns.std()
        return mu, sigma

    def _simulate_gbm_path(self, n_steps: int, s0: float | None = None) -> np.ndarray:
        """Generate one synthetic price path via GBM, calibrated to the
        real data's estimated drift/volatility."""
        if s0 is None:
            s0 = float(self.rng.choice(self.prices))  # start from a plausible real price level

        # GBM: S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)
        shocks = self.rng.normal(
            loc=(self.mu_daily - 0.5 * self.sigma_daily ** 2),
            scale=self.sigma_daily,
            size=n_steps,
        )
        log_path = np.cumsum(shocks)
        path = s0 * np.exp(log_path)
        return np.concatenate([[s0], path])  # include starting price

    def _sample_real_window(self, n_steps: int) -> np.ndarray:
        """Pick a random contiguous window of real historical prices."""
        max_start = len(self.prices) - (n_steps + 1)
        if max_start <= 0:
            # not enough history for this length -- fall back to GBM
            return self._simulate_gbm_path(n_steps)
        start = self.rng.integers(0, max_start)
        return self.prices[start:start + n_steps + 1]

    def get_episode_path(self, episode_length: int = DEFAULT_EPISODE_LENGTH, mix_real_prob: float = MIX_REAL_PROB) -> np.ndarray:
        """Return one price path (length episode_length+1, including the
        starting price) for a single training episode."""
        use_real = self.rng.random() < mix_real_prob
        if use_real:
            return self._sample_real_window(episode_length)
        return self._simulate_gbm_path(episode_length)


def main():
    price_path = os.path.join(RAW_DIR, "eth_daily.csv")
    sim = PriceSimulator(price_path, seed=42)

    print(f"Calibrated GBM from real data:")
    print(f"  daily drift (mu):       {sim.mu_daily:.6f}")
    print(f"  daily volatility (sig): {sim.sigma_daily:.6f}")
    print(f"  annualized drift:       {sim.mu_daily * 365:.4f}")
    print(f"  annualized volatility:  {sim.sigma_daily * np.sqrt(365):.4f}")

    print(f"\nGenerating 5 sample episodes (length={DEFAULT_EPISODE_LENGTH} days):")
    for i in range(5):
        path = sim.get_episode_path()
        pct_change = (path[-1] / path[0] - 1) * 100
        print(f"  episode {i+1}: start=${path[0]:.2f}  end=${path[-1]:.2f}  change={pct_change:+.1f}%")


if __name__ == "__main__":
    main()