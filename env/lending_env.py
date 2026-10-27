"""
lending_env.py  (v3 -- proposal reward)

Gymnasium environment for the DQN agent -- simulates a single borrower
position on an Aave-style lending pool across a mixed real/GBM ETH price
path, with the Risk Signal R in the state and a Credit Score CS shaping
the reward.

What changed in v3 (to match the thesis proposal, Sec 5.1):
  1. REWARD = proposal formula:
        reward = ALPHA * m(CS) * (change in net wealth)  -  BETA if liquidated
     "Change in net wealth" here is the part the agent's choices control:
     interest earned/paid + any liquidation loss. The ETH price move is left
     out of the reward because no action can change the user's ETH exposure
     (see 2), so it is the same for every strategy and would only add noise.
     The full net worth (incl. price moves) is still reported as the
     primary metric N_user.
     Credit multiplier m(CS) is asymmetric:
        gains  x (0.5 + CS)   -> good-history users are rewarded more
        losses x (1.5 - CS)   -> poor-history users are punished more
     R is no longer in the reward -- it is in the STATE, as an early warning
     (proposal Objective 2).
  2. WALLET PER ASSET: the user holds ETH and USDC in a wallet. Actions move
     tokens between wallet, collateral and debt without converting them, so
     the user's net ETH exposure never changes -- the agent manages the LOAN,
     it does not trade ETH.
  3. INTEREST (fixed yearly rates, accrued daily):
        borrow rate on debt, supply rate on collateral, and an external yield
        on idle wallet USDC (5% = Auto.gov's external competing supply rate).
     Borrowing to hold USDC earns a small spread, but adds liquidation risk.
     Putting USDC into collateral or repaying is safer but earns less.
     That trade-off is what the agent has to learn.
  4. SAFETY CONSTRAINT: a borrow (or withdraw) that would push HF below 1.05
     is blocked (proposal Sec 5.1). Blocked actions count as invalid.
  5. AAVE-STYLE LIQUIDATION when HF < 1: a liquidator repays 50% of the debt
     and takes that value + a 5% bonus from the collateral. Since v3.2 the
     episode CONTINUES after a liquidation (Aave liquidations are partial),
     so all strategies are valued on the same final day. Before v3.2 the
     episode ended, which froze a liquidated user's net worth at an earlier,
     higher ETH price and made liquidation look good in falling markets.
     info["liquidation_count"] counts liquidations in the episode.
  6. INVALID ACTION PENALTY: a tiny penalty for actions that do nothing
     (e.g. deposit with an empty wallet), so the agent picks "hold" instead.
  7. ACTION MASK (v3.1): info["action_mask"] says which actions can do
     something in the current state, so the DQN only chooses among those
     (like Aave's app greying out actions you can't take).
     ALPHA raised to 10 and BETA to 50 after the first tuning runs.

Unchanged from v2: ablation masking (R parts / CS zeroed in the observation
when switched off), normalized observation, train/test price split.

Action space: Discrete(N_ASSETS * 4 + 1)
  action // 4 = asset index (0=ETH, 1=USDC), action % 4 = type
  (0=deposit, 1=withdraw, 2=borrow, 3=repay); the last index = hold.

Observation (12 values):
  [wallet_ETH, wallet_USDC, collat_ETH, collat_USDC, debt_ETH, debt_USDC,
   (all in USD / 1000)
   eth_price / start_price, hf_scaled,
   health_deficit, volatility_norm, utilization, credit_score]

Usage (standalone checks):
    python env/lending_env.py
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

ASSETS = ["ETH", "USDC"]
N_ASSETS = len(ASSETS)
ACTION_TYPES = ["deposit", "withdraw", "borrow", "repay"]
HOLD_ACTION = N_ASSETS * len(ACTION_TYPES)  # last index = hold

LIQUIDATION_THRESHOLD = 0.8  # standard Aave-style assumption
ACTION_SIZE_USD = 100.0      # fixed size per deposit/withdraw/borrow/repay action
EPISODE_LENGTH = 90          # days per episode
VOL_WINDOW = 30              # rolling window for in-episode volatility
DAYS_PER_YEAR = 365

# ---- Starting position ----
START_COLLATERAL_ETH_USD = 1000.0  # $1000 of ETH as collateral
START_DEBT_USDC = 400.0            # $400 USDC borrowed -> HF = 2.0
START_WALLET_USDC = 400.0          # the borrowed $400 sits in the wallet

# ---- Interest rates (per year, accrued daily) -- assumptions, tunable ----
BORROW_APR = {"ETH": 0.03, "USDC": 0.03}   # cost of debt
SUPPLY_APR = {"ETH": 0.02, "USDC": 0.02}   # earned on collateral
EXTERNAL_YIELD_APR = 0.05                  # earned on idle wallet USDC (Auto.gov's external supply rate)

# ---- Safety constraint and liquidation (Aave-style) ----
MIN_HF_AFTER_ACTION = 1.05   # proposal: block borrows that would push H below 1.05
CLOSE_FACTOR = 0.5           # liquidator repays 50% of the debt
LIQUIDATION_BONUS = 0.05     # ...and takes 5% extra collateral as a bonus

# ---- Reward (proposal: alpha and beta are tunable) ----
ALPHA = 10.0                  # reward per USD of controllable wealth change (x10 so daily interest is "loud" enough to learn from)
BETA = 50.0                   # hard penalty on liquidation
INVALID_ACTION_PENALTY = 0.01

# Risk level above which a step counts as "high risk" for the proactive
# response rate metric (used by training_loop / evaluation). With the
# early-warning health deficit, R > 0.5 means roughly HF < 1.3.
SAFE_RISK_THRESHOLD = 0.5

# ---- Observation scaling ----
USD_SCALE = 1000.0
HF_CLIP = 5.0
EPS = 1e-9


class LendingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        price_csv_path: str = None,
        wallet_scores_csv_path: str = None,
        seed: int | None = None,
        use_risk_signal: bool = True,
        use_credit_score: bool = True,
        split: str = "train",   # "train" for training, "val" for tuning, "test" for final results
    ):
        super().__init__()

        price_csv_path = price_csv_path or os.path.join(DATA_DIR, "raw", "eth_daily.csv")
        wallet_scores_csv_path = wallet_scores_csv_path or os.path.join(DATA_DIR, "processed", "wallet_credit_scores.csv")

        self.split = split
        self.price_sim = PriceSimulator(price_csv_path, seed=seed, split=split)
        self.wallet_scores = pd.read_csv(wallet_scores_csv_path)
        self.rng = np.random.default_rng(seed)

        # ablation switches -- leave both True for the full model
        self.use_risk_signal = use_risk_signal
        self.use_credit_score = use_credit_score

        self.action_space = spaces.Discrete(HOLD_ACTION + 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        # episode state (token amounts), set in reset()
        self.price_path = None
        self.t = 0
        self.start_price = 1.0
        self.wallet = {a: 0.0 for a in ASSETS}
        self.collateral = {a: 0.0 for a in ASSETS}
        self.debt = {a: 0.0 for a in ASSETS}
        self.credit_score = 0.5
        self.liquidation_count = 0

    # ---------- value helpers ----------
    def _price(self, asset: str) -> float:
        if asset == "ETH":
            return float(self.price_path[self.t])
        return 1.0  # USDC pegged

    def _usd(self, balances: dict) -> float:
        return sum(balances[a] * self._price(a) for a in ASSETS)

    def _net_worth(self) -> float:
        return self._usd(self.wallet) + self._usd(self.collateral) - self._usd(self.debt)

    @staticmethod
    def _health_factor(collateral_usd: float, debt_usd: float) -> float:
        if debt_usd <= EPS:
            return 100.0  # "no risk" sentinel when there is no debt
        return (collateral_usd * LIQUIDATION_THRESHOLD) / debt_usd

    def _current_hf(self) -> float:
        return self._health_factor(self._usd(self.collateral), self._usd(self.debt))

    def _volatility_norm(self) -> float:
        window_start = max(0, self.t - VOL_WINDOW)
        window = self.price_path[window_start:self.t + 1]
        if len(window) < 2:
            return 0.0
        log_returns = np.diff(np.log(window))
        vol = log_returns.std() * np.sqrt(365)
        # rough fixed-range normalization (ETH annualized vol typically 0.2-1.2); clip to [0,1]
        return float(np.clip(vol / 1.2, 0.0, 1.0))

    def _risk_components(self):
        collateral_usd, debt_usd = self._usd(self.collateral), self._usd(self.debt)
        hf = self._health_factor(collateral_usd, debt_usd)
        health_deficit = compute_health_deficit(hf)
        volatility_norm = self._volatility_norm()
        utilization = compute_utilization(debt_usd, collateral_usd)
        return collateral_usd, debt_usd, hf, health_deficit, volatility_norm, utilization

    # ---------- observation ----------
    def _build_observation(self) -> np.ndarray:
        _, _, hf, health_deficit, volatility_norm, utilization = self._risk_components()
        p = self._price("ETH")

        # ablation masking: hide what this model variant is not supposed to see
        if not self.use_risk_signal:
            health_deficit = volatility_norm = utilization = 0.0
        cs = self.credit_score if self.use_credit_score else 0.0

        return np.array([
            self.wallet["ETH"] * p / USD_SCALE,
            self.wallet["USDC"] / USD_SCALE,
            self.collateral["ETH"] * p / USD_SCALE,
            self.collateral["USDC"] / USD_SCALE,
            self.debt["ETH"] * p / USD_SCALE,
            self.debt["USDC"] / USD_SCALE,
            p / self.start_price,
            min(hf, HF_CLIP) / HF_CLIP,
            health_deficit,
            volatility_norm,
            utilization,
            cs,
        ], dtype=np.float32)

    # ---------- actions ----------
    def _apply_action(self, action: int) -> bool:
        """Move tokens between wallet, collateral and debt.
        Returns False if the action did nothing (empty wallet, no debt, or
        blocked by the HF >= 1.05 safety constraint)."""
        if action == HOLD_ACTION:
            return True

        asset = ASSETS[action // len(ACTION_TYPES)]
        action_type = ACTION_TYPES[action % len(ACTION_TYPES)]
        size = ACTION_SIZE_USD / self._price(asset)  # token amount for a $100 action

        if action_type == "deposit":
            amount = min(size, self.wallet[asset])
            if amount <= EPS:
                return False
            self.wallet[asset] -= amount
            self.collateral[asset] += amount

        elif action_type == "withdraw":
            amount = min(size, self.collateral[asset])
            if amount <= EPS:
                return False
            self.collateral[asset] -= amount
            self.wallet[asset] += amount
            if self._usd(self.debt) > EPS and self._current_hf() < MIN_HF_AFTER_ACTION:
                self.collateral[asset] += amount  # undo -- unsafe
                self.wallet[asset] -= amount
                return False

        elif action_type == "borrow":
            self.debt[asset] += size
            self.wallet[asset] += size
            if self._current_hf() < MIN_HF_AFTER_ACTION:
                self.debt[asset] -= size  # undo -- unsafe
                self.wallet[asset] -= size
                return False

        elif action_type == "repay":
            amount = min(size, self.wallet[asset], self.debt[asset])
            if amount <= EPS:
                return False
            self.wallet[asset] -= amount
            self.debt[asset] -= amount

        return True

    def action_mask(self) -> np.ndarray:
        """Which actions would actually do something right now (True = allowed).
        Tries each action on the current balances, then restores them.
        "hold" is always allowed. Used for action masking in the DQN."""
        saved = (dict(self.wallet), dict(self.collateral), dict(self.debt))
        mask = np.zeros(self.action_space.n, dtype=bool)
        for a in range(self.action_space.n):
            mask[a] = self._apply_action(a)
            self.wallet, self.collateral, self.debt = dict(saved[0]), dict(saved[1]), dict(saved[2])
        return mask

    # ---------- interest and liquidation ----------
    def _accrue_interest(self) -> float:
        """One day of interest. Returns the USD change in net worth."""
        before = self._net_worth()
        self.wallet["USDC"] *= 1 + EXTERNAL_YIELD_APR / DAYS_PER_YEAR
        for a in ASSETS:
            self.collateral[a] *= 1 + SUPPLY_APR[a] / DAYS_PER_YEAR
            self.debt[a] *= 1 + BORROW_APR[a] / DAYS_PER_YEAR
        return self._net_worth() - before

    def _liquidate(self) -> float:
        """Aave-style: the liquidator repays CLOSE_FACTOR of the debt and
        takes that value + LIQUIDATION_BONUS from the collateral.
        Returns the USD change in net worth (negative = the bonus lost)."""
        before = self._net_worth()
        collateral_usd, debt_usd = self._usd(self.collateral), self._usd(self.debt)
        repaid_usd = CLOSE_FACTOR * debt_usd
        seized_usd = min(repaid_usd * (1 + LIQUIDATION_BONUS), collateral_usd)
        seized_frac = seized_usd / collateral_usd if collateral_usd > EPS else 0.0
        for a in ASSETS:
            self.debt[a] *= (1 - CLOSE_FACTOR)
            self.collateral[a] *= (1 - seized_frac)
        return self._net_worth() - before

    # ---------- reward ----------
    def _credit_multiplier(self, wealth_change: float) -> float:
        if not self.use_credit_score:
            return 1.0
        if wealth_change >= 0:
            return 0.5 + self.credit_score   # gains count more for good-history users
        return 1.5 - self.credit_score       # losses count more for poor-history users

    def _compute_reward(self, wealth_change: float, liquidated: bool, valid: bool) -> float:
        """Proposal reward: ALPHA * m(CS) * wealth change - BETA if liquidated,
        plus a tiny penalty for actions that did nothing."""
        reward = ALPHA * self._credit_multiplier(wealth_change) * wealth_change
        if liquidated:
            reward -= BETA
        if not valid:
            reward -= INVALID_ACTION_PENALTY
        return reward

    # ---------- gym API ----------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.price_path = self.price_sim.get_episode_path(episode_length=EPISODE_LENGTH)
        self.t = 0
        self.start_price = self._price("ETH")

        self.wallet = {"ETH": 0.0, "USDC": START_WALLET_USDC}
        self.collateral = {"ETH": START_COLLATERAL_ETH_USD / self.start_price, "USDC": 0.0}
        self.debt = {"ETH": 0.0, "USDC": START_DEBT_USDC}

        sampled_wallet = self.wallet_scores.sample(1, random_state=self.rng.integers(0, 1_000_000)).iloc[0]
        self.credit_score = float(sampled_wallet["credit_score"])
        self.liquidation_count = 0

        return self._build_observation(), self._info()

    def _info(self, **extra) -> dict:
        collateral_usd, debt_usd = self._usd(self.collateral), self._usd(self.debt)
        info = {
            "net_worth_usd": self._net_worth(),
            "wallet_usd": self._usd(self.wallet),
            "collateral_usd": collateral_usd,
            "debt_usd": debt_usd,
            "health_factor": self._health_factor(collateral_usd, debt_usd),
            "credit_score": self.credit_score,
            "action_mask": self.action_mask(),   # allowed actions in THIS state
        }
        info.update(extra)
        return info

    def step(self, action: int):
        valid = self._apply_action(int(action))

        self.t += 1                                # new day: price moves
        interest_usd = self._accrue_interest()

        _, debt_usd, hf, health_deficit, volatility_norm, utilization = self._risk_components()
        risk_signal = compute_risk_signal(health_deficit, volatility_norm, utilization)

        liquidated = hf < 1.0 and debt_usd > EPS
        liquidation_loss_usd = self._liquidate() if liquidated else 0.0
        self.liquidation_count += int(liquidated)

        wealth_change = interest_usd + liquidation_loss_usd   # the part actions control
        reward = self._compute_reward(wealth_change, liquidated, valid)

        # v3.2: a liquidation does NOT end the episode (Aave liquidations are
        # partial -- the position lives on). The episode always runs to day 90,
        # so every strategy's net worth is measured on the SAME final day.
        # Only a fully wiped-out collateral ends it early.
        terminated = bool(self._usd(self.collateral) <= EPS)
        truncated = bool(self.t >= EPISODE_LENGTH)

        info = self._info(
            health_factor=hf,                      # the HF that was checked this step
            risk_signal=risk_signal,
            health_deficit=health_deficit,
            volatility_norm=volatility_norm,
            utilization=utilization,
            liquidated=liquidated,                 # liquidated on THIS step
            liquidation_count=self.liquidation_count,  # liquidations so far this episode
            invalid_action=not valid,
            interest_usd=interest_usd,
            liquidation_loss_usd=liquidation_loss_usd,
        )
        return self._build_observation(), reward, terminated, truncated, info


# ---------- standalone checks ----------
def action_id(action_type: str, asset: str) -> int:
    return ASSETS.index(asset) * len(ACTION_TYPES) + ACTION_TYPES.index(action_type)


def heuristic_policy(info: dict) -> int:
    """Proposal baseline: repay when H < 1.2, borrow when H > 2.0, else hold."""
    if info["health_factor"] < 1.2:
        return action_id("repay", "USDC")
    if info["health_factor"] > 2.0:
        return action_id("borrow", "USDC")
    return HOLD_ACTION


def _run(env, policy_fn, label, reset=True):
    if reset:
        obs, info = env.reset()
    else:
        info = env._info()
    start_nw, start_price = info["net_worth_usd"], env.start_price
    total_reward = controllable = 0.0
    invalid = 0
    min_hf = info["health_factor"]
    while True:
        obs, reward, terminated, truncated, info = env.step(policy_fn(info))
        total_reward += reward
        controllable += info["interest_usd"] + info["liquidation_loss_usd"]
        invalid += int(info["invalid_action"])
        min_hf = min(min_hf, info["health_factor"])
        if terminated or truncated:
            break
    price_chg = (env._price("ETH") / start_price - 1) * 100
    print(f"[{label}] days={env.t:2d} liquidations={info['liquidation_count']} "
          f"N: {start_nw:.0f}->{info['net_worth_usd']:.0f} (ETH {price_chg:+.1f}%) "
          f"controllable={controllable:+.2f} min_HF={min_hf:.2f} invalid={invalid} reward={total_reward:.2f}")


def main():
    env = LendingEnv(seed=42)
    obs, info = env.reset()
    print("Initial observation (12 values):")
    print(np.round(obs, 3))
    print(f"Start: N=${info['net_worth_usd']:.0f} (wallet ${info['wallet_usd']:.0f} + collateral "
          f"${info['collateral_usd']:.0f} - debt ${info['debt_usd']:.0f}), HF={info['health_factor']:.2f}\n")

    # Check 1: three simple strategies run end to end
    _run(env, lambda i: env.action_space.sample(), "random   ")
    _run(env, lambda i: HOLD_ACTION, "static   ")
    _run(env, heuristic_policy, "heuristic")

    # Check 2: safety constraint -- borrowing every day must never push HF below 1.05
    _run(env, lambda i: action_id("borrow", "USDC"), "borrow-all")
    print("   -> min_HF should be >= 1.05 at the moment of borrowing; later blocked borrows count as invalid")

    # Check 3: forced crash -- ETH falls 60% in 90 days, static user gets liquidated
    # (the episode now continues to day 90 after a liquidation)
    env.reset()
    p0 = env.start_price
    env.price_path = np.linspace(p0, p0 * 0.4, len(env.price_path))
    _run(env, lambda i: HOLD_ACTION, "crash    ", reset=False)
    print(f"   -> should liquidate (>=1) and still run to day 90; each liquidation costs a "
          f"{LIQUIDATION_BONUS:.0%} bonus on {CLOSE_FACTOR:.0%} of the debt and -{BETA} reward")

    # Check 4: action mask matches what actually happens
    mismatches = 0
    for _ in range(5):
        obs, info = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            allowed = bool(info["action_mask"][a])
            obs, r, te, tr, info = env.step(a)
            mismatches += int(allowed == info["invalid_action"])
            done = te or tr
    print(f"\nAction mask check: {mismatches} mismatches (should be 0)")

    # Check 5: ablation masking hides R and CS from the observation
    env_masked = LendingEnv(seed=42, use_risk_signal=False, use_credit_score=False)
    obs_m, _ = env_masked.reset()
    print(f"\nMasked obs (last 4 should be 0): {np.round(obs_m[-4:], 3)}")


if __name__ == "__main__":
    main()