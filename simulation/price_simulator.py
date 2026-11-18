"""
price_simulator.py  (v4 -- calm / stormy markets + crash jumps, stress split)

What changed in v4 (6 Nov, upgrade plan Step 2):
  The new Risk Signal R uses VOLATILITY to say how fast danger is coming.
  That only helps if the market's speed actually CHANGES during training.
  v3's synthetic paths always moved at the same speed (one fixed sigma),
  so knowing the speed was useless.

  1. REGIME-SWITCHING GBM: each synthetic day is either CALM or STORMY.
        calm   : sigma = SIGMA_CALM_MULT  x train sigma  (0.6x)
        stormy : sigma = SIGMA_STORM_MULT x train sigma  (2.0x)
     The regime switches randomly (Markov chain):
        calm -> stormy with prob P_CALM_TO_STORM per day (1/30: calm lasts ~30 days)
        stormy -> calm with prob P_STORM_TO_CALM per day (1/10: storms last ~10 days)
     So about 25% of synthetic days are stormy (Hamilton 1989 style regimes).
  2. CRASH JUMPS: on any day, with prob JUMP_PROB (1/90, about one per
     episode), the price drops by 10-25% in one day.
  3. ZERO EXPECTED DRIFT is kept (like v3.1): a small daily compensation is
     added so that, on average, the price neither rises nor falls. No free trend.
  4. TRAINING mix 0.7 -> 0.5 real (MIX_REAL_PROB_TRAIN): half the training
     episodes are synthetic, so the agent sees enough storms and crashes.
     VAL / TEST keep the old 70% real / 30% synthetic mix (MIX_REAL_PROB_EVAL).
  5. NEW SPLIT "stress" (for evaluation only): every episode is synthetic,
     STARTS stormy and has 3x more crashes. Starting prices are taken from
     the VAL period. Use: python evaluate.py --split stress
  Real price windows are unchanged. GBM is still calibrated on TRAIN only.
  USE_REGIME_GBM = False gives the v3 single-sigma GBM back.

Usage:
    python simulation/price_simulator.py   # demo of all splits
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MIX_REAL_PROB_TRAIN = 0.5    # v4: was 0.7
MIX_REAL_PROB_EVAL = 0.7     # val / test: unchanged
DEFAULT_EPISODE_LENGTH = 90
TRAIN_FRAC = 0.7
VAL_FRAC = 0.1
SPLITS = ("train", "val", "test", "stress", "all")
GBM_ZERO_DRIFT = True

# ---- v4: regimes and jumps ----
USE_REGIME_GBM = True
SIGMA_CALM_MULT = 0.6
SIGMA_STORM_MULT = 2.0
P_CALM_TO_STORM = 1 / 30
P_STORM_TO_CALM = 1 / 10
JUMP_PROB = 1 / 90
JUMP_MIN, JUMP_MAX = 0.10, 0.25      # one-day crash size (price falls 10%-25%)
STRESS_JUMP_MULT = 3.0               # stress split: 3x more crashes, starts stormy


class PriceSimulator:
    def __init__(self, price_csv_path: str, seed: int | None = None,
                 split: str = "train", train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        df = pd.read_csv(price_csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        all_prices = df["close"].values
        all_dates = df["date"].values
        n = len(all_prices)
        train_end = int(round(n * train_frac))
        val_end = int(round(n * (train_frac + val_frac)))

        bounds = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, n),
            "stress": (train_end, val_end),   # start-price levels only (all paths synthetic)
            "all": (0, n),
        }
        start, end = bounds[split]

        self.split = split
        self.train_prices = all_prices[:train_end]
        self.prices = all_prices[start:end]
        self.dates = all_dates[start:end]
        self.rng = np.random.default_rng(seed)

        # GBM is always calibrated on TRAIN data only (no leakage)
        self.calibrated_mu_daily, self.sigma_daily = self._calibrate_gbm(self.train_prices)
        self.mu_daily = 0.0 if GBM_ZERO_DRIFT else self.calibrated_mu_daily

        self.jump_prob = JUMP_PROB * (STRESS_JUMP_MULT if split == "stress" else 1.0)
        self.last_regimes = None   # regimes of the last synthetic path (for checks), None for real

    @staticmethod
    def _calibrate_gbm(prices: np.ndarray) -> tuple[float, float]:
        log_returns = np.diff(np.log(prices))
        return log_returns.mean(), log_returns.std()

    def _simulate_gbm_path(self, n_steps: int, s0: float | None = None) -> np.ndarray:
        """v3 single-sigma GBM (used when USE_REGIME_GBM = False)."""
        if s0 is None:
            s0 = float(self.rng.choice(self.prices))
        shocks = self.rng.normal(loc=(self.mu_daily - 0.5 * self.sigma_daily ** 2),
                                 scale=self.sigma_daily, size=n_steps)
        self.last_regimes = np.zeros(n_steps, dtype=int)
        return np.concatenate([[s0], s0 * np.exp(np.cumsum(shocks))])

    def _jump_compensation(self) -> float:
        """Daily log-drift that cancels the expected effect of crash jumps,
        so E[price tomorrow / price today] = 1 (zero expected drift)."""
        u = np.linspace(JUMP_MIN, JUMP_MAX, 1001)          # crash sizes, uniform
        mean_factor = np.mean(1.0 - u)                     # E[e^J]
        return -np.log(1.0 - self.jump_prob + self.jump_prob * mean_factor)

    def _simulate_regime_path(self, n_steps: int, s0: float | None = None,
                              start_stormy: bool | None = None) -> np.ndarray:
        """v4: calm / stormy regimes + crash jumps, zero expected drift."""
        if s0 is None:
            s0 = float(self.rng.choice(self.prices))
        if start_stormy is None:
            stationary_storm = P_CALM_TO_STORM / (P_CALM_TO_STORM + P_STORM_TO_CALM)
            start_stormy = self.rng.random() < stationary_storm

        comp = self._jump_compensation() + self.mu_daily
        regimes = np.zeros(n_steps, dtype=int)
        log_ret = np.zeros(n_steps)
        stormy = bool(start_stormy)
        for t in range(n_steps):
            regimes[t] = int(stormy)
            sigma = self.sigma_daily * (SIGMA_STORM_MULT if stormy else SIGMA_CALM_MULT)
            r = self.rng.normal(-0.5 * sigma ** 2, sigma) + comp
            if self.rng.random() < self.jump_prob:
                r += np.log(1.0 - self.rng.uniform(JUMP_MIN, JUMP_MAX))
            log_ret[t] = r
            switch_p = P_STORM_TO_CALM if stormy else P_CALM_TO_STORM
            if self.rng.random() < switch_p:
                stormy = not stormy
        self.last_regimes = regimes
        return np.concatenate([[s0], s0 * np.exp(np.cumsum(log_ret))])

    def _synthetic_path(self, n_steps: int) -> np.ndarray:
        if self.split == "stress":
            return self._simulate_regime_path(n_steps, start_stormy=True)
        if USE_REGIME_GBM:
            return self._simulate_regime_path(n_steps)
        return self._simulate_gbm_path(n_steps)

    def _sample_real_window(self, n_steps: int) -> np.ndarray:
        max_start = len(self.prices) - (n_steps + 1)
        if max_start <= 0:
            return self._synthetic_path(n_steps)
        start = self.rng.integers(0, max_start + 1)
        self.last_regimes = None
        return self.prices[start:start + n_steps + 1]

    def get_episode_path(self, episode_length: int = DEFAULT_EPISODE_LENGTH,
                         mix_real_prob: float | None = None) -> np.ndarray:
        if mix_real_prob is None:
            mix_real_prob = MIX_REAL_PROB_TRAIN if self.split == "train" else MIX_REAL_PROB_EVAL
        if self.split == "stress":
            return self._synthetic_path(episode_length)
        if self.rng.random() < mix_real_prob:
            return self._sample_real_window(episode_length)
        return self._synthetic_path(episode_length)


def main():
    price_path = os.path.join(RAW_DIR, "eth_daily.csv")
    for split in ("train", "val", "test", "stress"):
        sim = PriceSimulator(price_path, seed=42, split=split)
        print(f"=== {split.upper()} split ===")
        print(f"  days in split: {len(sim.prices)}  ({str(sim.dates[0])[:10]} to {str(sim.dates[-1])[:10]})")
        print(f"  train daily sigma {sim.sigma_daily:.4f} -> calm {sim.sigma_daily * SIGMA_CALM_MULT:.4f}, "
              f"stormy {sim.sigma_daily * SIGMA_STORM_MULT:.4f}; crash prob/day {sim.jump_prob:.4f}")
        for i in range(3):
            path = sim.get_episode_path()
            kind = "real" if sim.last_regimes is None else f"synthetic, {sim.last_regimes.mean():.0%} stormy days"
            print(f"    episode {i + 1}: change {(path[-1] / path[0] - 1) * 100:+6.1f}%  ({kind})")
        print()

    # Check: zero expected drift and the stormy share of synthetic days
    sim = PriceSimulator(price_path, seed=0, split="train")
    ends, storm = [], []
    for _ in range(4000):
        p = sim._simulate_regime_path(DEFAULT_EPISODE_LENGTH)
        ends.append(p[-1] / p[0])
        storm.append(sim.last_regimes.mean())
    print(f"Check: mean price change over 90 days = {(np.mean(ends) - 1) * 100:+.1f}% (should be near 0), "
          f"stormy days = {np.mean(storm):.0%} (should be near 25%)")


if __name__ == "__main__":
    main()