"""
lending_env.py

Gymnasium environment for the DQN agent -- simulates a single borrower
position across a mixed real/GBM ETH price path, augmented with the
Risk Signal R components and a Credit Score CS drawn from real wallet
data.

Action space (per-asset, resolved per your decision over the 5-action
proposal wording): for each asset, 4 action types -- deposit, withdraw,
borrow, repay -- plus one shared hold action.
  Discrete(N_ASSETS * 4 + 1)
  action // 4 = asset index, action % 4 = action type (0=deposit,
  1=withdraw, 2=borrow, 3=repay), except the final index which is hold.

Reward (Step 6): -risk_penalty - liquidation_penalty + safety_bonus, where
risk_penalty = risk_signal * credit_score_modulator. See _compute_reward()
for the full breakdown and rationale.

The `use_risk_signal` / `use_credit_score` flags let you switch each
component on/off independently -- needed for the Step 10 ablation study
(all 4 combinations of R/CS presence) without touching step() again.

Usage (standalone smoke test):
    python lending_env.py
"""

import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulation"))

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))

from price_simulator import PriceSimulator  # noqa: E402
from compute_risk_signal import (  # noqa: E402
    compute_health_deficit,
    compute_utilization,
    compute_risk_signal,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

ASSETS = ["ETH", "USDC"]  # index 0 = collateral asset, index 1 = debt-side asset
N_ASSETS = len(ASSETS)
ACTION_TYPES = ["deposit", "withdraw", "borrow", "repay"]
HOLD_ACTION = N_ASSETS * len(ACTION_TYPES)  # last index = hold

LIQUIDATION_THRESHOLD = 0.8  # standard Aave-style assumption; verify against defisol later if needed
ACTION_SIZE_USD = 100.0      # fixed notional size per deposit/withdraw/borrow/repay action
EPISODE_LENGTH = 90          # days per episode
VOL_WINDOW = 30              # rolling window for in-episode volatility

# ---- Reward shaping (Step 6) ----
# CS modulator range: CS=1.0 (perfect history) -> 0.5x risk penalty (benefit of the doubt)
#                      CS=0.0 (no/bad history)  -> 1.5x risk penalty (held to a stricter standard)
CS_MODULATOR_BASE = 1.5
LIQUIDATION_PENALTY = 10.0   # dominant terminal penalty -- liquidation should swamp everything else
SAFE_RISK_THRESHOLD = 0.2    # below this, the position counts as "comfortably safe"
SAFETY_BONUS = 0.05          # small per-step reward for staying comfortably safe (proactive behavior)


class LendingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        price_csv_path: str = None,
        wallet_scores_csv_path: str = None,
        seed: int | None = None,
        use_risk_signal: bool = True,
        use_credit_score: bool = True,
    ):
        super().__init__()

        price_csv_path = price_csv_path or os.path.join(DATA_DIR, "raw", "eth_daily.csv")
        wallet_scores_csv_path = wallet_scores_csv_path or os.path.join(DATA_DIR, "processed", "wallet_credit_scores.csv")

        self.price_sim = PriceSimulator(price_csv_path, seed=seed)
        self.wallet_scores = pd.read_csv(wallet_scores_csv_path)
        self.rng = np.random.default_rng(seed)

        # Step 10 ablation switches -- leave both True for the full model.
        self.use_risk_signal = use_risk_signal
        self.use_credit_score = use_credit_score

        self.action_space = spaces.Discrete(HOLD_ACTION + 1)

        # obs: [collateral_bal_ETH, collateral_bal_USDC, debt_bal_ETH, debt_bal_USDC,
        #       eth_price, health_factor, health_deficit, volatility_norm, utilization, credit_score]
        obs_dim = 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # episode state, set in reset()
        self.price_path = None
        self.t = 0
        self.collateral_balance = {a: 0.0 for a in ASSETS}
        self.debt_balance = {a: 0.0 for a in ASSETS}
        self.credit_score = 0.5

    def _asset_price(self, asset: str) -> float:
        if asset == "ETH":
            return float(self.price_path[self.t])
        return 1.0  # USDC pegged

    def _portfolio_values(self) -> tuple[float, float]:
        collateral_usd = sum(self.collateral_balance[a] * self._asset_price(a) for a in ASSETS)
        debt_usd = sum(self.debt_balance[a] * self._asset_price(a) for a in ASSETS)
        return collateral_usd, debt_usd

    def _health_factor(self, collateral_usd: float, debt_usd: float) -> float:
        if debt_usd <= 0:
            return 100.0  # effectively "no risk" sentinel when no debt outstanding
        return (collateral_usd * LIQUIDATION_THRESHOLD) / debt_usd

    def _volatility_norm(self) -> float:
        window_start = max(0, self.t - VOL_WINDOW)
        window = self.price_path[window_start:self.t + 1]
        if len(window) < 2:
            return 0.0
        log_returns = np.diff(np.log(window))
        vol = log_returns.std() * np.sqrt(365)
        # rough fixed-range normalization (ETH annualized vol typically 0.2-1.2); clip to [0,1]
        return float(np.clip(vol / 1.2, 0.0, 1.0))

    def _build_observation(self) -> np.ndarray:
        collateral_usd, debt_usd = self._portfolio_values()
        hf = self._health_factor(collateral_usd, debt_usd)
        health_deficit = compute_health_deficit(hf)
        volatility_norm = self._volatility_norm()
        utilization = compute_utilization(debt_usd, collateral_usd)

        return np.array([
            self.collateral_balance["ETH"],
            self.collateral_balance["USDC"],
            self.debt_balance["ETH"],
            self.debt_balance["USDC"],
            self._asset_price("ETH"),
            hf,
            health_deficit,
            volatility_norm,
            utilization,
            self.credit_score,
        ], dtype=np.float32)

    def _compute_reward(self, risk_signal: float, liquidated: bool) -> float:
        """Step 6: R and CS jointly shape the reward.

        - risk_penalty: the raw risk_signal, scaled by a credit-score modulator.
          High CS (good repayment/liquidation history) -> smaller modulator ->
          less punishing for the same risk level ("benefit of the doubt").
          Low CS -> larger modulator -> the same risk level is punished harder,
          nudging historically-risky wallets toward more conservative behavior.
        - liquidation_penalty: a large, dominant terminal penalty so the agent
          never learns to trade a slightly better risk_penalty for a liquidation.
        - safety_bonus: small reward for staying comfortably below the risk
          threshold, not just for avoiding liquidation outright -- this is what
          should drive up the "proactive response rate" eval metric.

        use_risk_signal / use_credit_score let each component be switched off
        independently for the Step 10 ablation (4 combinations of R/CS presence).
        """
        r = risk_signal if self.use_risk_signal else 0.0
        cs_modulator = (CS_MODULATOR_BASE - self.credit_score) if self.use_credit_score else 1.0
        risk_penalty = r * cs_modulator

        reward = -risk_penalty

        if liquidated:
            reward -= LIQUIDATION_PENALTY

        if self.use_risk_signal and risk_signal < SAFE_RISK_THRESHOLD:
            reward += SAFETY_BONUS

        return reward

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.price_path = self.price_sim.get_episode_path(episode_length=EPISODE_LENGTH)
        self.t = 0

        # starting position: a modest, plausible collateralized loan
        start_eth_price = self._asset_price("ETH")
        self.collateral_balance = {"ETH": 1000.0 / start_eth_price, "USDC": 0.0}  # ~$1000 collateral
        self.debt_balance = {"ETH": 0.0, "USDC": 400.0}  # $400 debt -> ~40% starting LTV

        sampled_wallet = self.wallet_scores.sample(1, random_state=self.rng.integers(0, 1_000_000)).iloc[0]
        self.credit_score = float(sampled_wallet["credit_score"])

        obs = self._build_observation()
        info = {}
        return obs, info

    def step(self, action: int):
        if action != HOLD_ACTION:
            asset_idx = action // len(ACTION_TYPES)
            action_type = ACTION_TYPES[action % len(ACTION_TYPES)]
            asset = ASSETS[asset_idx]
            price = self._asset_price(asset)
            amount = ACTION_SIZE_USD / price

            if action_type == "deposit":
                self.collateral_balance[asset] += amount
            elif action_type == "withdraw":
                self.collateral_balance[asset] = max(0.0, self.collateral_balance[asset] - amount)
            elif action_type == "borrow":
                self.debt_balance[asset] += amount
            elif action_type == "repay":
                self.debt_balance[asset] = max(0.0, self.debt_balance[asset] - amount)

        self.t += 1

        collateral_usd, debt_usd = self._portfolio_values()
        hf = self._health_factor(collateral_usd, debt_usd)
        health_deficit = compute_health_deficit(hf)
        volatility_norm = self._volatility_norm()
        utilization = compute_utilization(debt_usd, collateral_usd)
        risk_signal = compute_risk_signal(health_deficit, volatility_norm, utilization)

        liquidated = hf < 1.0 and debt_usd > 0
        reward = self._compute_reward(risk_signal, liquidated)

        terminated = bool(liquidated)
        truncated = bool(self.t >= EPISODE_LENGTH)

        obs = self._build_observation()
        info = {
            "health_factor": hf,
            "risk_signal": risk_signal,
            "health_deficit": health_deficit,
            "volatility_norm": volatility_norm,
            "utilization": utilization,
            "credit_score": self.credit_score,
            "liquidated": liquidated,
            "collateral_usd": collateral_usd,
            "debt_usd": debt_usd,
            "net_worth_usd": collateral_usd - debt_usd,
        }
        return obs, reward, terminated, truncated, info


def main():
    env = LendingEnv(seed=42)
    obs, info = env.reset()
    print("Initial observation:", obs)
    print(f"Sampled credit score for this episode: {env.credit_score:.3f}")

    total_reward = 0.0
    for step_i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step_i % 5 == 0:
            print(f"step {step_i}: action={action} reward={reward:.4f} HF={info['health_factor']:.3f} liquidated={info['liquidated']}")
        if terminated or truncated:
            print(f"Episode ended at step {step_i} (terminated={terminated}, truncated={truncated})")
            break

    print(f"\nTotal reward over episode: {total_reward:.4f}")


if __name__ == "__main__":
    main()