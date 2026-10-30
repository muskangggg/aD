"""
evaluate.py  (v2 -- better measurement, fix plan step 1)

Runs every strategy on the SAME price paths (same seed -> same paths and
same credit scores for everyone, because actions never use the env's random
generator), with no exploration, and prints two tables.

Strategies:
  - static     : never rebalances (Auto.gov's static baseline)
  - heuristic  : repays when H < 1.2, borrows when H > 2.0, else holds
  - random     : random actions (sanity floor)
  - borrow_all : NEW in v2 -- always tries to borrow USDC; only the 1.05
                 safety rule stops it. Shows how much safety the rule gives
                 on its own ("rule-only" baseline).
Trained DQN runs are added with --runs, e.g. full_seed42 r_only_seed42.

TABLE 1 -- main results (proposal Sec 5.4), per episode then averaged:
  N_end      : net worth at episode end (primary metric)
  liq_rate   : share of episodes with at least one liquidation
  liq_count  : average liquidations per episode
  risk_adj   : N_end / (1 + liquidation count)
  ctrl_gain  : interest + liquidation loss (the part strategies control;
               the ETH price move is the same for every strategy)
  invalid    : share of actions that did nothing

TABLE 2 -- behaviour (NEW in v2). Step-based rates are counted over ALL steps
of ALL episodes of a strategy (pooled), so episodes with few risky days do
not distort them.
  proactive  : of high-risk steps, share where a deposit/repay worked
  normal     : the same rate in normal (not high-risk) steps
  pro_gap    : proactive - normal. Above 0 = the agent de-risks MORE when
               risk is high, i.e. it really reacts to risk. A strategy that
               presses deposit/repay all the time gets a gap near 0.
  high_risk  : share of steps that were high risk (R > SAFE_RISK_THRESHOLD)
  hf_lt_1.1  : share of steps with HF < 1.1 (very close to liquidation)
  int_earned : interest earned (wallet USDC 5% + collateral 2%), $ per episode
  int_paid   : interest paid on debt (3%), $ per episode
  end_debt   : debt at episode end, $ (starts at $400)
  idle_eth   : average ETH sitting in the wallet, $ (earns 0%)

v2 change to "proactive": risk is now measured BEFORE the action (the state
the strategy actually saw when it chose), not after the day's price move.
Example: on a risky day the agent repays $100 and HF goes back above 1.3.
v1 checked the risk AFTER the repay, saw "safe", and did not count that
good repay at all. v2 counts it. So v2 numbers differ a bit from v1.
R is always the TRUE risk, also for no-R variants that cannot see it.

Usage:
    python evaluate.py                                  # baselines only, VALIDATION split
    python evaluate.py --runs full_seed42               # + a trained model (validation)
    python evaluate.py --runs full_seed42 none_seed42 --episodes 300
    python evaluate.py --runs ... --split test          # FINAL results only, run once
Per-episode results are saved to results/eval_<split>.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))

from lending_env import (  # noqa: E402
    LendingEnv, ASSETS, ACTION_TYPES, HOLD_ACTION, SAFE_RISK_THRESHOLD,
    BORROW_APR, SUPPLY_APR, EXTERNAL_YIELD_APR, DAYS_PER_YEAR,
    heuristic_policy, action_id, compute_risk_signal,
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
EVAL_SEED = 2026   # fixed -> every strategy sees identical price paths
LOW_HF = 1.1       # "very close to liquidation" level for the hf_lt_1.1 column

VARIANT_FLAGS = {  # run-name prefix -> (use_risk_signal, use_credit_score)
    "full": (True, True),
    "r_only": (True, False),
    "cs_only": (False, True),
    "none": (False, False),
}
DE_RISKING = {"deposit", "repay"}


def is_de_risking(action: int) -> bool:
    return action != HOLD_ACTION and ACTION_TYPES[action % len(ACTION_TYPES)] in DE_RISKING


class EvalEnv(LendingEnv):
    """The same environment, but it also remembers how much interest was
    PAID and EARNED each day. The normal env only reports the net amount
    (earned - paid). Nothing about the simulation changes."""

    def _accrue_interest(self) -> float:
        p = {a: self._price(a) for a in ASSETS}
        self.last_paid = sum(self.debt[a] * p[a] * BORROW_APR[a] for a in ASSETS) / DAYS_PER_YEAR
        self.last_earned = (self.wallet["USDC"] * EXTERNAL_YIELD_APR
                            + sum(self.collateral[a] * p[a] * SUPPLY_APR[a] for a in ASSETS)) / DAYS_PER_YEAR
        return super()._accrue_interest()


def risk_before_action(env: LendingEnv) -> tuple[float, float]:
    """(HF, R) of the state the strategy sees right now, before it acts."""
    _, _, hf, health_deficit, volatility_norm, utilization = env._risk_components()
    return hf, compute_risk_signal(health_deficit, volatility_norm, utilization)


def make_dqn_policy(run_name: str, n_obs: int, n_actions: int):
    import torch  # only needed for trained models
    from dqn_agent import QNetwork, masked_argmax

    net = QNetwork(n_obs, n_actions)
    net.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, run_name, "final.pt"), map_location="cpu"))
    net.eval()

    def act(obs, info):
        with torch.no_grad():
            q = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        return masked_argmax(q, info["action_mask"])
    return act


def evaluate(name, policy, use_r, use_cs, split, episodes):
    env = EvalEnv(seed=EVAL_SEED, split=split, use_risk_signal=use_r, use_credit_score=use_cs)
    rows = []
    for _ in range(episodes):
        obs, info = env.reset()
        c = dict(steps=0, invalid=0, high_steps=0, high_derisk=0, normal_steps=0, normal_derisk=0,
                 low_hf_steps=0)
        ctrl = paid = earned = idle_eth = 0.0
        done = False
        while not done:
            hf_now, r_now = risk_before_action(env)       # what the strategy sees
            idle_eth += env.wallet["ETH"] * env._price("ETH")
            a = policy(obs, info)
            obs, reward, terminated, truncated, info = env.step(a)

            worked_derisk = is_de_risking(a) and not info["invalid_action"]
            c["steps"] += 1
            c["invalid"] += int(info["invalid_action"])
            c["low_hf_steps"] += int(hf_now < LOW_HF)
            if r_now > SAFE_RISK_THRESHOLD:
                c["high_steps"] += 1
                c["high_derisk"] += int(worked_derisk)
            else:
                c["normal_steps"] += 1
                c["normal_derisk"] += int(worked_derisk)
            ctrl += info["interest_usd"] + info["liquidation_loss_usd"]
            paid += env.last_paid
            earned += env.last_earned
            done = terminated or truncated

        liq_count = int(info["liquidation_count"])
        n_end = info["net_worth_usd"]
        rows.append({
            "strategy": name,
            "N_end": n_end,
            "liquidated": liq_count > 0,
            "liq_count": liq_count,
            "risk_adj": n_end / (1 + liq_count),
            "ctrl_gain": ctrl,
            "int_earned": earned,
            "int_paid": paid,
            "end_debt": info["debt_usd"],
            "idle_eth": idle_eth / c["steps"],
            **c,
        })
    return pd.DataFrame(rows)


def safe_div(a, b):
    return a / b if b else np.nan


def summarise(results: pd.DataFrame, order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = results.groupby("strategy", sort=False)

    main_table = g.agg(N_end=("N_end", "mean"), liq_rate=("liquidated", "mean"),
                       liq_count=("liq_count", "mean"), risk_adj=("risk_adj", "mean"),
                       ctrl_gain=("ctrl_gain", "mean")).reindex(order)
    sums = g[["steps", "invalid", "high_steps", "high_derisk", "normal_steps", "normal_derisk",
              "low_hf_steps"]].sum().reindex(order)
    main_table["invalid"] = sums["invalid"] / sums["steps"]

    behaviour = pd.DataFrame(index=order)
    behaviour.index.name = "strategy"
    behaviour["proactive"] = [safe_div(h, n) for h, n in zip(sums["high_derisk"], sums["high_steps"])]
    behaviour["normal"] = [safe_div(h, n) for h, n in zip(sums["normal_derisk"], sums["normal_steps"])]
    behaviour["pro_gap"] = behaviour["proactive"] - behaviour["normal"]
    behaviour["high_risk"] = sums["high_steps"] / sums["steps"]
    behaviour[f"hf_lt_{LOW_HF}"] = sums["low_hf_steps"] / sums["steps"]
    means = g[["int_earned", "int_paid", "end_debt", "idle_eth"]].mean().reindex(order)
    behaviour = behaviour.join(means)
    return main_table, behaviour


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[], help="trained run names, e.g. full_seed42")
    p.add_argument("--split", default="val", choices=["train", "val", "test"],
                   help="val (default) for tuning; test ONLY for the final results")
    p.add_argument("--episodes", type=int, default=200)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    probe = LendingEnv(seed=EVAL_SEED, split=args.split)
    n_obs, n_actions = probe.observation_space.shape[0], probe.action_space.n
    borrow_usdc = action_id("borrow", "USDC")

    strategies = [
        ("static", lambda o, i: HOLD_ACTION, True, True),
        ("heuristic", lambda o, i: heuristic_policy(i), True, True),
        ("random", lambda o, i: int(rng.integers(0, n_actions)), True, True),
        ("borrow_all", lambda o, i: borrow_usdc, True, True),
    ]
    for run in args.runs:
        use_r, use_cs = VARIANT_FLAGS[run.rsplit("_seed", 1)[0]]
        strategies.append((run, make_dqn_policy(run, n_obs, n_actions), use_r, use_cs))

    if args.split == "test":
        print("!!! TEST split: use this only for the FINAL results, not for tuning !!!\n")
    print(f"=== Evaluation on {args.split.upper()} split, {args.episodes} episodes each, "
          f"same price paths for every strategy ===\n")
    results = pd.concat([evaluate(n, pol, r, cs, args.split, args.episodes)
                         for n, pol, r, cs in strategies], ignore_index=True)

    order = [s[0] for s in strategies]
    main_table, behaviour = summarise(results, order)

    pd.set_option("display.width", 160)
    print("TABLE 1 - main results")
    print(main_table.round({"N_end": 1, "liq_rate": 3, "liq_count": 2, "risk_adj": 1,
                            "ctrl_gain": 2, "invalid": 3}).to_string())
    print(f"\nTABLE 2 - behaviour (high risk = R > {SAFE_RISK_THRESHOLD}, risk measured before acting)")
    print(behaviour.round(3).round({"int_earned": 2, "int_paid": 2, "end_debt": 0, "idle_eth": 0}).to_string())
    print("\nHow to read: pro_gap > 0 means it de-risks MORE when risk is high (real reaction to risk).")
    print("ctrl_gain = int_earned - int_paid + liquidation loss.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"eval_{args.split}.csv")
    results.to_csv(out, index=False)
    print(f"\nPer-episode results saved to {out}")


if __name__ == "__main__":
    main()