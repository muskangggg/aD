"""
inspect_policy.py

Diagnostic: load a trained model and watch what it actually does, with
NO random exploration (epsilon = 0), on the TEST price split.

Prints:
  1. A day-by-day trace of the first episode (action, valid?, HF, cash, debt)
  2. How often each action is chosen (valid vs. invalid)
  3. A summary over all episodes: liquidations, ending debt / cash,
     and on which day the loan was fully repaid (if ever)

Usage:
    python inspect_policy.py                      # full_seed42, test split
    python inspect_policy.py --run r_only_seed42
    python inspect_policy.py --split train --episodes 50
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))

from lending_env import LendingEnv, ACTION_TYPES, ASSETS, HOLD_ACTION  # noqa: E402
from dqn_agent import QNetwork, masked_argmax  # noqa: E402

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

VARIANT_FLAGS = {  # run-name prefix -> (use_risk_signal, use_credit_score)
    "full": (True, True),
    "r_only": (True, False),
    "cs_only": (False, True),
    "none": (False, False),
}


def action_name(a: int) -> str:
    if a == HOLD_ACTION:
        return "hold"
    return f"{ACTION_TYPES[a % len(ACTION_TYPES)]} {ASSETS[a // len(ACTION_TYPES)]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="full_seed42")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--trace-days", type=int, default=15)
    args = p.parse_args()

    prefix = args.run.rsplit("_seed", 1)[0]
    use_r, use_cs = VARIANT_FLAGS[prefix]

    env = LendingEnv(seed=123, split=args.split, use_risk_signal=use_r, use_credit_score=use_cs)
    net = QNetwork(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, args.run, "final.pt"), map_location="cpu"))
    net.eval()

    def act(obs, mask):
        with torch.no_grad():
            q = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        return masked_argmax(q, mask)

    print(f"=== {args.run} on {args.split.upper()} split, {args.episodes} episodes, epsilon=0 ===\n")

    valid_counts, invalid_counts = Counter(), Counter()
    liquidations, end_debts, end_cash, end_wealth_chg, repaid_day = [], [], [], [], []

    for ep in range(args.episodes):
        obs, info = env.reset()
        start_nw = info["net_worth_usd"]
        first_zero_debt = None

        if ep == 0:
            print(f"Episode 1 trace (first {args.trace_days} days):")
            print(f"  {'day':>3}  {'action':<14} {'valid':<6} {'HF':>7} {'wallet':>7} {'collat':>7} {'debt':>7}")

        done = False
        while not done:
            a = act(obs, info["action_mask"])
            obs, reward, terminated, truncated, info = env.step(a)
            (invalid_counts if info["invalid_action"] else valid_counts)[action_name(a)] += 1

            if first_zero_debt is None and info["debt_usd"] <= 1e-6:
                first_zero_debt = env.t

            if ep == 0 and env.t <= args.trace_days:
                hf = info["health_factor"]
                hf_str = "no debt" if hf >= 100 else f"{hf:.2f}"
                print(f"  {env.t:>3}  {action_name(a):<14} {str(not info['invalid_action']):<6} "
                      f"{hf_str:>7} {info['wallet_usd']:>7.0f} {info['collateral_usd']:>7.0f} {info['debt_usd']:>7.0f}")

            done = terminated or truncated

        liquidations.append(info["liquidated"])
        end_debts.append(info["debt_usd"])
        end_cash.append(info["wallet_usd"])
        end_wealth_chg.append(info["net_worth_usd"] - start_nw)
        repaid_day.append(first_zero_debt)

    total = sum(valid_counts.values()) + sum(invalid_counts.values())
    print(f"\nAction choices over all episodes ({total} steps):")
    print(f"  {'action':<14} {'valid':>7} {'invalid':>8} {'share':>7}")
    for name in sorted(set(valid_counts) | set(invalid_counts),
                       key=lambda n: -(valid_counts[n] + invalid_counts[n])):
        n = valid_counts[name] + invalid_counts[name]
        print(f"  {name:<14} {valid_counts[name]:>7} {invalid_counts[name]:>8} {n / total:>7.1%}")

    repaid = [d for d in repaid_day if d is not None]
    print(f"\nSummary:")
    print(f"  liquidation rate:          {np.mean(liquidations):.2f}")
    print(f"  avg net wealth change:     {np.mean(end_wealth_chg):.2f}")
    print(f"  avg ending debt:           ${np.mean(end_debts):.0f}  (started at $400)")
    print(f"  avg ending wallet:         ${np.mean(end_cash):.0f}  (started at $400)")
    print(f"  episodes where loan fully repaid: {len(repaid)}/{len(repaid_day)}"
          + (f", on average by day {np.mean(repaid):.1f}" if repaid else ""))


if __name__ == "__main__":
    main()