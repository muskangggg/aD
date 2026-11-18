"""
lending_env.py  (v4 -- credit-based loan terms + forward-looking Risk Signal)

Gymnasium environment for the DQN agent -- simulates a single borrower
position on an Aave-style lending pool across a mixed real/synthetic ETH
price path, with the Risk Signal R in the state and the Credit Score CS
setting the user's loan terms.

What changed in v4 (6 Nov, upgrade plan, approved by the supervisor):
  Step 1 (cs_analysis.py) and the 30 Oct ablation showed that neither R
  nor CS changed what the BEST action was, so agents that could see them
  had no reason to use them:
    - R v2 was mostly HF rescaled (HF is already in the observation).
    - CS only re-weighted the reward (m(CS)), while every user had the same
      loan. The best plan was "stay safe" for everyone, so CS changed
      nothing except adding reward noise.
  Rule we follow now: a feature can only help if it changes WHAT THE BEST
  ACTION IS.

  1. CREDIT-BASED BORROW RATE (USE_CS_LOAN_TERMS): CS sets the user's rate,
     like a bank gives cheaper loans to customers with a good history:
         borrow APR = 2% + 6% x (1 - CS)
         CS 0.25 -> 6.5%   CS 0.50 -> 5.0%   CS 0.95 -> 2.3%
     (break-even with the 5% wallet yield at CS 0.5, close to the median user,
     so about half the users should borrow and half should repay).
     Idle wallet USDC still earns 5%. So for a high-CS user borrowing to
     hold USDC EARNS a spread; for a low-CS user it LOSES money and repaying
     is better. The best action now depends on CS.
  2. NO CS REWARD WEIGHT (USE_CS_REWARD_WEIGHT = False): reward is now
         ALPHA * (interest + liquidation loss) - BETA per liquidation
     The old m(CS) multiplier (0.5+CS gains / 1.5-CS losses) is kept in the
     code but off. CS now acts through the loan, not through the reward.
  3. FORWARD-LOOKING RISK SIGNAL R (compute_risk_signal v3):
         R = P(liquidation within 5 days | position, today's volatility)
     Volatility is a RiskMetrics EWMA (lambda 0.94) updated every day, so
     it reacts within a few days when the market turns stormy.
     Observation slots 8-10 are now: R, volatility (EWMA, annualized / 2),
     utilization. "High risk" (SAFE_RISK_THRESHOLD) is now R > 5%.
  4. ABLATION = HIDING INFORMATION ONLY:
         --no-r  : R, volatility and utilization are zeroed in the observation
         --no-cs : CS is zeroed in the observation
     The WORLD stays the same for every variant (the user still pays their
     CS-based rate, the market is still calm or stormy). So any difference
     between variants comes only from what the agent can SEE.
  Unchanged: actions, $100 action size, 1.05 safety rule, Aave-style
  liquidation (50% close factor, 5% bonus), episodes run to day 90,
  optional danger-zone penalty (off), observation size 12.

Action space: Discrete(9)
  action // 4 = asset (0=ETH, 1=USDC), action % 4 = type
  (0=deposit, 1=withdraw, 2=borrow, 3=repay); 8 = hold.

Observation (12 values):
  [wallet_ETH, wallet_USDC, collat_ETH, collat_USDC, debt_ETH, debt_USDC,
   (all in USD / 1000)
   eth_price / start_price, hf_scaled,
   R (liquidation probability, 5 days), volatility_norm, utilization, credit_score]

Usage (standalone checks):
    python env/lending_env.py
"""

import math
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
    compute_liquidation_probability,
    liquidation_distance,
    update_ewma_variance,
    RISK_HORIZON_DAYS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

ASSETS = ["ETH", "USDC"]
N_ASSETS = len(ASSETS)
ACTION_TYPES = ["deposit", "withdraw", "borrow", "repay"]
HOLD_ACTION = N_ASSETS * len(ACTION_TYPES)

LIQUIDATION_THRESHOLD = 0.8
ACTION_SIZE_USD = 100.0
EPISODE_LENGTH = 90
DAYS_PER_YEAR = 365

# ---- Starting position ----
START_COLLATERAL_ETH_USD = 1000.0
START_DEBT_USDC = 400.0
START_WALLET_USDC = 400.0

# ---- Interest rates (per year, accrued daily) ----
BORROW_APR = {"ETH": 0.03, "USDC": 0.03}   # v3 rate for everyone (used when USE_CS_LOAN_TERMS = False)
SUPPLY_APR = {"ETH": 0.02, "USDC": 0.02}
EXTERNAL_YIELD_APR = 0.05

# ---- v4: credit-based loan terms ----
USE_CS_LOAN_TERMS = True
CS_RATE_BASE = 0.02            # borrow APR = CS_RATE_BASE + CS_RATE_SPREAD * (1 - CS)
CS_RATE_SPREAD = 0.06
USE_CS_REWARD_WEIGHT = False   # v3 m(CS) reward multiplier, off in v4

# ---- Safety constraint and liquidation ----
MIN_HF_AFTER_ACTION = 1.05
CLOSE_FACTOR = 0.5
LIQUIDATION_BONUS = 0.05

# ---- Reward ----
ALPHA = 10.0
BETA = 50.0
INVALID_ACTION_PENALTY = 0.01

# ---- optional HF danger-zone penalty (v3.3 experiment, OFF) ----
HF_BUFFER = 1.15
DEFAULT_BUFFER_PENALTY = 0.0

# v4: a step is "high risk" when the chance of liquidation within
# RISK_HORIZON_DAYS is above 5% (used by the proactive / pro_gap metrics).
SAFE_RISK_THRESHOLD = 0.05

# ---- Observation scaling ----
USD_SCALE = 1000.0
HF_CLIP = 5.0
VOL_NORM_SCALE = 2.0     # annualized volatility / 2 (stormy ETH ~1.3 -> 0.65)
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
        split: str = "train",   # train / val / test / stress
        buffer_penalty: float = DEFAULT_BUFFER_PENALTY,
    ):
        super().__init__()

        price_csv_path = price_csv_path or os.path.join(DATA_DIR, "raw", "eth_daily.csv")
        wallet_scores_csv_path = wallet_scores_csv_path or os.path.join(DATA_DIR, "processed", "wallet_credit_scores.csv")

        self.split = split
        self.price_sim = PriceSimulator(price_csv_path, seed=seed, split=split)
        self.wallet_scores = pd.read_csv(wallet_scores_csv_path)
        self.rng = np.random.default_rng(seed)

        # ablation switches: they only hide information from the observation
        self.use_risk_signal = use_risk_signal
        self.use_credit_score = use_credit_score
        self.buffer_penalty = float(buffer_penalty)

        self.action_space = spaces.Discrete(HOLD_ACTION + 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.price_path = None
        self.t = 0
        self.start_price = 1.0
        self.wallet = {a: 0.0 for a in ASSETS}
        self.collateral = {a: 0.0 for a in ASSETS}
        self.debt = {a: 0.0 for a in ASSETS}
        self.credit_score = 0.5
        self.liquidation_count = 0
        self.var_daily = self.price_sim.sigma_daily ** 2   # EWMA variance (v4)

    # ---------- value helpers ----------
    def _price(self, asset: str) -> float:
        if asset == "ETH":
            return float(self.price_path[self.t])
        return 1.0

    def _usd(self, balances: dict) -> float:
        return sum(balances[a] * self._price(a) for a in ASSETS)

    def _net_worth(self) -> float:
        return self._usd(self.wallet) + self._usd(self.collateral) - self._usd(self.debt)

    @staticmethod
    def _health_factor(collateral_usd: float, debt_usd: float) -> float:
        if debt_usd <= EPS:
            return 100.0
        return (collateral_usd * LIQUIDATION_THRESHOLD) / debt_usd

    def _current_hf(self) -> float:
        return self._health_factor(self._usd(self.collateral), self._usd(self.debt))

    # ---------- v4: loan terms ----------
    def borrow_apr(self, asset: str) -> float:
        """The user's yearly borrow rate. v4: depends on the credit score."""
        if not USE_CS_LOAN_TERMS:
            return BORROW_APR[asset]
        return CS_RATE_BASE + CS_RATE_SPREAD * (1.0 - self.credit_score)

    # ---------- v4: volatility and risk ----------
    def sigma_daily(self) -> float:
        return math.sqrt(max(self.var_daily, 0.0))

    def _volatility_norm(self) -> float:
        return float(np.clip(self.sigma_daily() * math.sqrt(DAYS_PER_YEAR) / VOL_NORM_SCALE, 0.0, 1.0))

    def _liquidation_probability(self) -> float:
        p = self._price("ETH")
        d, already = liquidation_distance(
            self.collateral["ETH"] * p, self.collateral["USDC"],
            self.debt["ETH"] * p, self.debt["USDC"], LIQUIDATION_THRESHOLD)
        return compute_liquidation_probability(d, already, self.sigma_daily(), RISK_HORIZON_DAYS)

    def _risk_components(self):
        """(collateral_usd, debt_usd, hf, R, volatility_norm, utilization).
        v4: the 4th value is the liquidation probability R (was health_deficit),
        so older scripts calling compute_risk_signal(x4, x5, x6) still get R."""
        collateral_usd, debt_usd = self._usd(self.collateral), self._usd(self.debt)
        hf = self._health_factor(collateral_usd, debt_usd)
        liq_prob = self._liquidation_probability()
        volatility_norm = self._volatility_norm()
        utilization = compute_utilization(debt_usd, collateral_usd)
        return collateral_usd, debt_usd, hf, liq_prob, volatility_norm, utilization

    # ---------- observation ----------
    def _build_observation(self) -> np.ndarray:
        _, _, hf, liq_prob, volatility_norm, utilization = self._risk_components()
        p = self._price("ETH")

        if not self.use_risk_signal:
            liq_prob = volatility_norm = utilization = 0.0
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
            liq_prob,
            volatility_norm,
            utilization,
            cs,
        ], dtype=np.float32)

    # ---------- actions ----------
    def _apply_action(self, action: int) -> bool:
        if action == HOLD_ACTION:
            return True

        asset = ASSETS[action // len(ACTION_TYPES)]
        action_type = ACTION_TYPES[action % len(ACTION_TYPES)]
        size = ACTION_SIZE_USD / self._price(asset)

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
                self.collateral[asset] += amount
                self.wallet[asset] -= amount
                return False

        elif action_type == "borrow":
            self.debt[asset] += size
            self.wallet[asset] += size
            if self._current_hf() < MIN_HF_AFTER_ACTION:
                self.debt[asset] -= size
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
        saved = (dict(self.wallet), dict(self.collateral), dict(self.debt))
        mask = np.zeros(self.action_space.n, dtype=bool)
        for a in range(self.action_space.n):
            mask[a] = self._apply_action(a)
            self.wallet, self.collateral, self.debt = dict(saved[0]), dict(saved[1]), dict(saved[2])
        return mask

    # ---------- interest and liquidation ----------
    def _accrue_interest(self) -> float:
        before = self._net_worth()
        self.wallet["USDC"] *= 1 + EXTERNAL_YIELD_APR / DAYS_PER_YEAR
        for a in ASSETS:
            self.collateral[a] *= 1 + SUPPLY_APR[a] / DAYS_PER_YEAR
            self.debt[a] *= 1 + self.borrow_apr(a) / DAYS_PER_YEAR
        return self._net_worth() - before

    def _liquidate(self) -> float:
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
        """v3 m(CS) weight. v4: off (USE_CS_REWARD_WEIGHT = False) -> always 1."""
        if not USE_CS_REWARD_WEIGHT or not self.use_credit_score:
            return 1.0
        if wealth_change >= 0:
            return 0.5 + self.credit_score
        return 1.5 - self.credit_score

    def _buffer_penalty(self, hf: float) -> float:
        if self.buffer_penalty <= 0 or hf >= HF_BUFFER:
            return 0.0
        return self.buffer_penalty * (HF_BUFFER - hf) / (HF_BUFFER - MIN_HF_AFTER_ACTION)

    def _compute_reward(self, wealth_change: float, liquidated: bool, valid: bool,
                        buffer_penalty: float = 0.0) -> float:
        reward = ALPHA * self._credit_multiplier(wealth_change) * wealth_change
        if liquidated:
            reward -= BETA
        if not valid:
            reward -= INVALID_ACTION_PENALTY
        reward -= buffer_penalty
        return reward

    # ---------- gym API ----------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.price_path = self.price_sim.get_episode_path(episode_length=EPISODE_LENGTH)
        self.t = 0
        self.start_price = self._price("ETH")
        self.var_daily = self.price_sim.sigma_daily ** 2   # start from the train-calibrated level

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
            "borrow_apr_usdc": self.borrow_apr("USDC"),   # v4: the user's CS-based rate
            "action_mask": self.action_mask(),
        }
        info.update(extra)
        return info

    def step(self, action: int):
        valid = self._apply_action(int(action))

        self.t += 1                                 # new day: price moves
        log_ret = math.log(self.price_path[self.t] / self.price_path[self.t - 1])
        self.var_daily = update_ewma_variance(self.var_daily, log_ret)   # v4: volatility update
        interest_usd = self._accrue_interest()

        _, debt_usd, hf, liq_prob, volatility_norm, utilization = self._risk_components()
        risk_signal = compute_risk_signal(liq_prob, volatility_norm, utilization)

        liquidated = hf < 1.0 and debt_usd > EPS
        liquidation_loss_usd = self._liquidate() if liquidated else 0.0
        self.liquidation_count += int(liquidated)

        wealth_change = interest_usd + liquidation_loss_usd
        buffer_pen = self._buffer_penalty(hf)
        reward = self._compute_reward(wealth_change, liquidated, valid, buffer_pen)

        terminated = bool(self._usd(self.collateral) <= EPS)
        truncated = bool(self.t >= EPISODE_LENGTH)

        info = self._info(
            health_factor=hf,
            risk_signal=risk_signal,                 # v4: liquidation probability (5 days)
            health_deficit=compute_health_deficit(hf),   # kept for logs only
            volatility_norm=volatility_norm,
            utilization=utilization,
            liquidated=liquidated,
            liquidation_count=self.liquidation_count,
            invalid_action=not valid,
            interest_usd=interest_usd,
            liquidation_loss_usd=liquidation_loss_usd,
            buffer_penalty=buffer_pen,
        )
        return self._build_observation(), reward, terminated, truncated, info


# ---------- baselines and standalone checks ----------
def action_id(action_type: str, asset: str) -> int:
    return ASSETS.index(asset) * len(ACTION_TYPES) + ACTION_TYPES.index(action_type)


def heuristic_policy(info: dict) -> int:
    """Proposal baseline: repay when H < 1.2, borrow when H > 2.0, else hold (CS-blind)."""
    if info["health_factor"] < 1.2:
        return action_id("repay", "USDC")
    if info["health_factor"] > 2.0:
        return action_id("borrow", "USDC")
    return HOLD_ACTION


def heuristic_cs_policy(info: dict) -> int:
    """v4 stronger baseline that KNOWS the CS rule: same as the heuristic, but
    when the user's borrow rate is above the 5% wallet yield, borrowing loses
    money, so it never borrows and pays the debt down instead."""
    mask = info["action_mask"]
    repay = action_id("repay", "USDC")
    if info["health_factor"] < 1.2:
        return repay
    if info["borrow_apr_usdc"] > EXTERNAL_YIELD_APR:
        return repay if mask[repay] else HOLD_ACTION
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
    max_r = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy_fn(info))
        total_reward += reward
        controllable += info["interest_usd"] + info["liquidation_loss_usd"]
        invalid += int(info["invalid_action"])
        min_hf = min(min_hf, info["health_factor"])
        max_r = max(max_r, info["risk_signal"])
        if terminated or truncated:
            break
    price_chg = (env._price("ETH") / start_price - 1) * 100
    print(f"[{label}] CS={env.credit_score:.2f} rate={env.borrow_apr('USDC'):.1%} "
          f"liquidations={info['liquidation_count']} N: {start_nw:.0f}->{info['net_worth_usd']:.0f} "
          f"(ETH {price_chg:+.1f}%) controllable={controllable:+.2f} min_HF={min_hf:.2f} "
          f"max_R={max_r:.2f} invalid={invalid} reward={total_reward:.2f}")
    return controllable


def main():
    env = LendingEnv(seed=42)
    obs, info = env.reset()
    print("Initial observation (12 values):")
    print(np.round(obs, 3))
    print(f"Start: N=${info['net_worth_usd']:.0f}, HF={info['health_factor']:.2f}, "
          f"CS={env.credit_score:.2f}, borrow rate={env.borrow_apr('USDC'):.2%}\n")

    # Check 1: strategies run end to end
    _run(env, lambda i: env.action_space.sample(), "random    ")
    _run(env, lambda i: HOLD_ACTION, "static    ")
    _run(env, heuristic_policy, "heuristic ")
    _run(env, heuristic_cs_policy, "heur_cs   ")
    _run(env, lambda i: action_id("borrow", "USDC"), "borrow-all")

    # Check 2: forced crash -> liquidation, episode continues to day 90
    env.reset()
    p0 = env.start_price
    env.price_path = np.linspace(p0, p0 * 0.4, len(env.price_path))
    _run(env, lambda i: HOLD_ACTION, "crash     ", reset=False)
    print("   -> should liquidate (>=1), still run to day 90, and max_R should reach 1.00")

    # Check 3: action mask matches what happens
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
    print(f"\nCheck 3 - action mask: {mismatches} mismatches (should be 0)")

    # Check 4: ablation masking (R, vol, util, CS hidden), the world unchanged
    env_m = LendingEnv(seed=42, use_risk_signal=False, use_credit_score=False)
    obs_m, info_m = env_m.reset()
    env_f = LendingEnv(seed=42)
    obs_f, info_f = env_f.reset()
    same_world = (env_m.credit_score == env_f.credit_score and
                  abs(env_m.borrow_apr("USDC") - env_f.borrow_apr("USDC")) < 1e-12)
    print(f"Check 4 - masked obs (last 4 should be 0): {np.round(obs_m[-4:], 3)}; "
          f"same user and rate as the full env: {same_world} (should be True)")

    # Check 5: CS sets the borrow rate (the whole point of v4)
    print("Check 5 - borrow rate by CS (wallet USDC earns 5%):")
    for cs in (0.25, 0.50, 0.75, 0.95):
        env.credit_score = cs
        rate = env.borrow_apr("USDC")
        verdict = "borrowing EARNS" if rate < EXTERNAL_YIELD_APR else "borrowing LOSES" \
            if rate > EXTERNAL_YIELD_APR else "break-even"
        print(f"   CS {cs:.2f} -> {rate:.2%}  ({verdict})")

    # Check 6: same position, low CS vs high CS: holding vs repaying the $400 debt
    print("Check 6 - 90 days, same prices, static vs repay-all, low vs high CS:")
    for cs in (0.25, 0.95):
        res = {}
        for name, pol in (("static", lambda i: HOLD_ACTION),
                          ("repay-all", lambda i: action_id("repay", "USDC")
                           if i["action_mask"][action_id("repay", "USDC")] else HOLD_ACTION)):
            e = LendingEnv(seed=7)
            e.reset()
            e.credit_score = cs
            e.price_path = np.full(len(e.price_path), e.start_price)   # flat price: only interest matters
            gain = 0.0
            info = e._info()
            while True:
                _, _, te, tr, info = e.step(pol(info))
                gain += info["interest_usd"]
                if te or tr:
                    break
            res[name] = gain
        better = "repay-all" if res["repay-all"] > res["static"] else "static"
        print(f"   CS {cs:.2f}: static {res['static']:+.2f}$, repay-all {res['repay-all']:+.2f}$ -> best: {better}")
    print("   -> should be: low CS -> repay-all better, high CS -> static (keep the loan) better")

    # Check 7: R reacts to volatility at the same HF
    env.reset()
    env.debt = {"ETH": 0.0, "USDC": env._usd(env.collateral) * LIQUIDATION_THRESHOLD / 1.3}
    env.var_daily = 0.035 ** 2
    r_calm = env._liquidation_probability()
    env.var_daily = 0.07 ** 2
    r_storm = env._liquidation_probability()
    print(f"Check 7 - HF 1.30: R calm (3.5%/day) = {r_calm:.2%}, R stormy (7%/day) = {r_storm:.2%} "
          f"(stormy should be much higher)")


if __name__ == "__main__":
    main()