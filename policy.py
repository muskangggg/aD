"""
policy.py  (v1.0 -- live decision function for the Sepolia service, with the R shield with memory)

What this file is:
  The bridge between the trained DQN agent (left part: data, R, CS, offline
  training) and the live DeFi service (right part). Once a day the service
  gives it the user's position, today's ETH price and the user's Credit
  Score; it answers with ONE action (0-8) and the amount to send.

It makes decisions EXACTLY like evaluate.py v3.5 does in the simulator:
  - same 12-value observation (USD / 1000, price / start price, HF / 5,
    R, volatility, utilization, CS), with R / volatility / utilization
    hidden for no-R agents and CS hidden for no-CS agents
  - same R = P(liquidation within 5 days), volatility = RiskMetrics EWMA
    (lambda 0.94), updated once per new daily price
  - same action mask (an action is allowed only if it does something and
    keeps HF >= 1.05 after a borrow / withdraw)
  - same SHIELD WITH MEMORY (thesis part 3, sections 8-9b):
        ON  when the TRUE R > 5%                   -> borrow and withdraw blocked
        OFF only when R <= 1% AND >= 5 days since R was last above 5%
    Example: after a storm warning the parent keeps the car keys until the
    sky is really clear AND the warning is 5 days old.
  test_policy.py checks this: same actions as evaluate.py on every step.

Default agent: cs_only_seed42_env4 (deployment agent: 0 liquidations on
VAL, STRESS and TEST, with and without the shield).

How the live service uses it (one call per day, in this order):
    from policy import LivePolicy
    pol = LivePolicy()                               # loads checkpoints/<run>/final.pt
    pol.warm_up(last_60_daily_eth_closes)            # optional, once: sets today's volatility
    d = pol.decide(eth_price, position, credit_score)
    # position = {"wallet_ETH": .., "wallet_USDC": .., "collat_ETH": .., "collat_USDC": ..,
    #             "debt_ETH": .., "debt_USDC": ..}   (TOKEN amounts, not USD)
    # d.action (0-8), d.action_name ("repay_USDC"), d.asset, d.amount (tokens), d.R, d.shield_on
    if d.action != 8: send d.action_name with d.amount to the pool

Two live details the simulator did not have:
  1. Volatility needs a starting value. Training started every episode from
     the TRAIN daily sigma. Live, call warm_up() with recent real daily
     closes so R reflects today's market (otherwise the TRAIN sigma is used,
     read from data/raw/eth_daily.csv, or pass sigma_daily_init).
  2. Observation slot 6 is price / start price of a 90-day episode. Live,
     the start price is the first price seen and is reset every 90 days
     (EPISODE_LENGTH), like a new episode. The shield memory is NOT reset.
  Call decide() only ONCE per new daily price (or pass new_day=False to ask
  again on the same day without moving the volatility or the shield clock).
"""

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "env"))
sys.path.append(os.path.join(HERE, "features"))
sys.path.append(os.path.join(HERE, "simulation"))

from compute_risk_signal import (  # noqa: E402
    compute_utilization,
    compute_liquidation_probability,
    liquidation_distance,
    update_ewma_variance,
    RISK_HORIZON_DAYS,
)

# ---- must match env/lending_env.py v4.1 (test_policy.py Check A compares them) ----
ASSETS = ["ETH", "USDC"]
ACTION_TYPES = ["deposit", "withdraw", "borrow", "repay"]
HOLD_ACTION = len(ASSETS) * len(ACTION_TYPES)          # 8
N_ACTIONS = HOLD_ACTION + 1                            # 9
N_OBS = 12
LIQUIDATION_THRESHOLD = 0.8
ACTION_SIZE_USD = 100.0
MIN_HF_AFTER_ACTION = 1.05
EPISODE_LENGTH = 90
DAYS_PER_YEAR = 365
USD_SCALE = 1000.0
HF_CLIP = 5.0
VOL_NORM_SCALE = 2.0
EPS = 1e-9

# ---- shield with memory (frozen 20 Nov; confirmed STRESS 2026 + 4049, TEST no cost) ----
SHIELD_ON_R = 0.05
SHIELD_OFF_R = 0.01
SHIELD_DAYS = 5
RE_RISKING = {"borrow", "withdraw"}

DEFAULT_RUN = "cs_only_seed42_env4"
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints")
PRICE_CSV = os.path.join(HERE, "data", "raw", "eth_daily.csv")

VARIANT_FLAGS = {            # (sees R, sees CS) -- same as evaluate.py
    "full": (True, True),
    "r_only": (True, False),
    "cs_only": (False, True),
    "none": (False, False),
}
POSITION_KEYS = ("wallet_ETH", "wallet_USDC", "collat_ETH", "collat_USDC", "debt_ETH", "debt_USDC")


def action_name(action: int) -> str:
    """0 -> 'deposit_ETH', 6 -> 'borrow_USDC', 8 -> 'hold' (same as evaluate.py)."""
    if action == HOLD_ACTION:
        return "hold"
    return f"{ACTION_TYPES[action % len(ACTION_TYPES)]}_{ASSETS[action // len(ACTION_TYPES)]}"


def shield_mask(mask: np.ndarray) -> np.ndarray:
    """Shield ON -> block borrow / withdraw (both assets). Same as evaluate.py."""
    m = np.array(mask, dtype=bool)
    for a in range(len(m)):
        if a != HOLD_ACTION and ACTION_TYPES[a % len(ACTION_TYPES)] in RE_RISKING:
            m[a] = False
    return m


@dataclass
class Decision:
    action: int                 # 0-8 (8 = hold)
    action_name: str            # e.g. "repay_USDC"
    asset: str | None           # "ETH" / "USDC" / None for hold
    action_type: str | None     # deposit / withdraw / borrow / repay / None
    amount: float               # token amount to send (0 for hold)
    R: float                    # TRUE R before acting (also for agents that cannot see R)
    health_factor: float
    shield_on: bool
    changed_by_shield: bool     # the agent's own first choice was blocked by the shield
    q_values: list = field(default_factory=list)
    mask: list = field(default_factory=list)
    obs: list = field(default_factory=list)


class LivePolicy:
    def __init__(self, run: str = DEFAULT_RUN, checkpoint_dir: str = CHECKPOINT_DIR,
                 shield: float | None = SHIELD_ON_R, shield_off: float | None = SHIELD_OFF_R,
                 shield_days: int = SHIELD_DAYS, sigma_daily_init: float | None = None,
                 q_function=None):
        """run: checkpoint folder name; its prefix (full / r_only / cs_only / none) sets
        what the agent can see. shield=None turns the shield off (not recommended).
        q_function: only for tests -- a function obs -> 9 Q-values instead of the network."""
        self.run = run
        self.use_r, self.use_cs = VARIANT_FLAGS[run.rsplit("_seed", 1)[0]]
        self.shield = shield
        self.shield_off = shield if shield_off is None else shield_off
        self.shield_days = shield_days
        self.q_function = q_function or self._load_network(os.path.join(checkpoint_dir, run, "final.pt"))

        if sigma_daily_init is None:
            sigma_daily_init = self._train_sigma()
        self.sigma_daily_init = float(sigma_daily_init)
        self.reset()

    # ---------- setup ----------
    @staticmethod
    def _load_network(path: str):
        import torch
        from dqn_agent import QNetwork

        net = QNetwork(N_OBS, N_ACTIONS)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()

        def q(obs):
            with torch.no_grad():
                return net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        return q

    @staticmethod
    def _train_sigma() -> float:
        """Daily sigma of TRAIN prices = the value every training episode started from."""
        if not os.path.exists(PRICE_CSV):
            raise FileNotFoundError(
                f"{PRICE_CSV} not found: pass sigma_daily_init=... or copy data/raw/eth_daily.csv")
        from price_simulator import PriceSimulator
        return float(PriceSimulator(PRICE_CSV, seed=0, split="train").sigma_daily)

    def reset(self, start_price: float | None = None, sigma_daily: float | None = None):
        """Start fresh (new user / new position): volatility, start price, shield memory.
        Tests call this at the start of every simulated episode."""
        self.var_daily = (self.sigma_daily_init if sigma_daily is None else float(sigma_daily)) ** 2
        self.start_price = start_price
        self.last_price = None            # day 1 of an episode: no return yet (like the env at t = 0)
        self.episode_day = 0
        self.day = 0
        self.shield_on = False
        self.last_alarm = -10 ** 9

    def warm_up(self, daily_closes):
        """Run the EWMA volatility over recent real daily ETH closes (oldest first).
        Call once before the first decide(); the last close becomes 'yesterday'."""
        closes = [float(p) for p in daily_closes]
        for prev, cur in zip(closes[:-1], closes[1:]):
            self.var_daily = update_ewma_variance(self.var_daily, math.log(cur / prev))
        if closes:
            self.last_price = closes[-1]

    # ---------- the same maths as LendingEnv ----------
    @staticmethod
    def _usd(bal: dict, price: float) -> float:
        return bal["ETH"] * price + bal["USDC"]

    @staticmethod
    def _hf(coll_usd: float, debt_usd: float) -> float:
        if debt_usd <= EPS:
            return 100.0
        return coll_usd * LIQUIDATION_THRESHOLD / debt_usd

    def _risk(self, pos: dict, price: float):
        coll_usd, debt_usd = self._usd(pos["collat"], price), self._usd(pos["debt"], price)
        hf = self._hf(coll_usd, debt_usd)
        sigma = math.sqrt(max(self.var_daily, 0.0))
        d, already = liquidation_distance(pos["collat"]["ETH"] * price, pos["collat"]["USDC"],
                                          pos["debt"]["ETH"] * price, pos["debt"]["USDC"],
                                          LIQUIDATION_THRESHOLD)
        r = compute_liquidation_probability(d, already, sigma, RISK_HORIZON_DAYS)
        vol_norm = float(np.clip(sigma * math.sqrt(DAYS_PER_YEAR) / VOL_NORM_SCALE, 0.0, 1.0))
        util = compute_utilization(debt_usd, coll_usd)
        return coll_usd, debt_usd, hf, r, vol_norm, util

    def _apply(self, pos: dict, action: int, price: float):
        """Copy of LendingEnv._apply_action on a copy of the position.
        Returns (valid, token amount moved)."""
        if action == HOLD_ACTION:
            return True, 0.0
        w, c, dbt = pos["wallet"], pos["collat"], pos["debt"]
        asset = ASSETS[action // len(ACTION_TYPES)]
        kind = ACTION_TYPES[action % len(ACTION_TYPES)]
        p = price if asset == "ETH" else 1.0
        size = ACTION_SIZE_USD / p

        if kind == "deposit":
            amount = min(size, w[asset])
            return amount > EPS, max(amount, 0.0)
        if kind == "withdraw":
            amount = min(size, c[asset])
            if amount <= EPS:
                return False, 0.0
            c2 = dict(c)
            c2[asset] -= amount
            debt_usd = self._usd(dbt, price)
            if debt_usd > EPS and self._hf(self._usd(c2, price), debt_usd) < MIN_HF_AFTER_ACTION:
                return False, 0.0
            return True, amount
        if kind == "borrow":
            d2 = dict(dbt)
            d2[asset] += size
            if self._hf(self._usd(c, price), self._usd(d2, price)) < MIN_HF_AFTER_ACTION:
                return False, 0.0
            return True, size
        # repay
        amount = min(size, w[asset], dbt[asset])
        return amount > EPS, max(amount, 0.0)

    def action_mask(self, pos: dict, price: float) -> np.ndarray:
        return np.array([self._apply(pos, a, price)[0] for a in range(N_ACTIONS)], dtype=bool)

    def _observation(self, pos, price, credit_score, r, vol_norm, util, hf) -> np.ndarray:
        if not self.use_r:
            r_o = vol_o = util_o = 0.0
        else:
            r_o, vol_o, util_o = r, vol_norm, util
        cs = credit_score if self.use_cs else 0.0
        return np.array([
            pos["wallet"]["ETH"] * price / USD_SCALE,
            pos["wallet"]["USDC"] / USD_SCALE,
            pos["collat"]["ETH"] * price / USD_SCALE,
            pos["collat"]["USDC"] / USD_SCALE,
            pos["debt"]["ETH"] * price / USD_SCALE,
            pos["debt"]["USDC"] / USD_SCALE,
            price / self.start_price,
            min(hf, HF_CLIP) / HF_CLIP,
            r_o, vol_o, util_o, cs,
        ], dtype=np.float32)

    # ---------- shield memory (same order as evaluate.py v3.5) ----------
    def _update_shield(self, r: float) -> bool:
        if self.shield is not None:
            if r > self.shield:                                      # alarm: switch ON (or stay ON)
                self.shield_on, self.last_alarm = True, self.day
            elif self.shield_on and r <= self.shield_off and self.day - self.last_alarm >= self.shield_days:
                self.shield_on = False                               # sky clear AND warning old enough
        self.day += 1
        return self.shield_on

    # ---------- main call ----------
    @staticmethod
    def _parse_position(position: dict) -> dict:
        missing = [k for k in POSITION_KEYS if k not in position]
        if missing:
            raise KeyError(f"position is missing {missing}; needs {POSITION_KEYS} (token amounts)")
        return {
            "wallet": {"ETH": float(position["wallet_ETH"]), "USDC": float(position["wallet_USDC"])},
            "collat": {"ETH": float(position["collat_ETH"]), "USDC": float(position["collat_USDC"])},
            "debt": {"ETH": float(position["debt_ETH"]), "USDC": float(position["debt_USDC"])},
        }

    def decide(self, eth_price: float, position: dict, credit_score: float, new_day: bool = True) -> Decision:
        price = float(eth_price)
        pos = self._parse_position(position)

        if new_day:
            if self.last_price is not None:                          # yesterday known -> today's return
                self.var_daily = update_ewma_variance(self.var_daily, math.log(price / self.last_price))
            if self.start_price is None or self.episode_day >= EPISODE_LENGTH:
                self.start_price, self.episode_day = price, 0        # new 90-day "episode"
            self.episode_day += 1
            self.last_price = price

        _, _, hf, r, vol_norm, util = self._risk(pos, price)
        mask = self.action_mask(pos, price)
        obs = self._observation(pos, price, float(credit_score), r, vol_norm, util, hf)
        q = np.asarray(self.q_function(obs), dtype=np.float64)

        shield_on = self._update_shield(r) if new_day else self.shield_on
        own_choice = _masked_argmax(q, mask)
        action = _masked_argmax(q, shield_mask(mask)) if shield_on else own_choice

        _, amount = self._apply(pos, action, price)
        name = action_name(action)
        asset = None if action == HOLD_ACTION else ASSETS[action // len(ACTION_TYPES)]
        kind = None if action == HOLD_ACTION else ACTION_TYPES[action % len(ACTION_TYPES)]
        return Decision(action, name, asset, kind, float(amount), float(r), float(hf), bool(shield_on),
                        bool(action != own_choice), q.tolist(), mask.tolist(), obs.tolist())


def _masked_argmax(q_values, mask) -> int:
    """Same as dqn_agent.masked_argmax (kept here so the live service needs no training code)."""
    q = np.array(q_values, dtype=np.float64)
    q[~np.asarray(mask, dtype=bool)] = -np.inf
    return int(np.argmax(q))


if __name__ == "__main__":
    # tiny demo: one decision for the standard starting position
    pol = LivePolicy()
    price = 3000.0
    pos = {"wallet_ETH": 0.0, "wallet_USDC": 400.0, "collat_ETH": 1000.0 / price,
           "collat_USDC": 0.0, "debt_ETH": 0.0, "debt_USDC": 400.0}
    for cs in (0.25, 0.95):
        pol.reset()
        d = pol.decide(price, pos, cs)
        print(f"CS {cs:.2f}: action {d.action} ({d.action_name}, {d.amount:.4f} {d.asset or ''}) "
              f"HF {d.health_factor:.2f} R {d.R:.2%} shield {'ON' if d.shield_on else 'off'}")