"""
evaluate.py  (Step 4 + 5: baselines and evaluation)

Runs every strategy on the SAME test price paths (same seed -> same paths
and same credit scores for everyone, because actions never use the random
generator), with no exploration, and prints one comparison table with the
proposal's 4 metrics (Sec 5.4) plus helpers.

Baselines (proposal Sec 5.4):
  - static    : never rebalances (Auto.gov's static baseline)
  - heuristic : repays when H < 1.2, borrows when H > 2.0, else holds
  - random    : random actions (sanity floor)
Trained DQN runs are added with --runs, e.g. full_seed42 r_only_seed42.

Metrics (per episode, then averaged):
  - N_end          : user net worth at episode end (primary metric)
  - liq_rate       : share of episodes ending in liquidation
  - risk_adj       : N_end / (1 + liquidation_count)
  - proactive      : of high-risk steps (R > SAFE_RISK_THRESHOLD), share where a
                     deposit or repay was made that actually worked
  - ctrl_gain      : interest + liquidation loss (the part strategies control;
                     the ETH price move is identical for every strategy)
  - invalid        : share of actions that did nothing

Usage:
    python evaluate.py                                  # baselines only
    python evaluate.py --runs full_seed42               # + a trained model
    python evaluate.py --runs full_seed42 none_seed42 --episodes 300
Results are also saved to results/eval_<split>.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))

from lending_env import (  # noqa: E402
    LendingEnv, ACTION_TYPES, HOLD_ACTION, SAFE_RISK_THRESHOLD,
    heuristic_policy,
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
EVAL_SEED = 2026  # fixed -> every strategy sees identical price paths

VARIANT_FLAGS = {  # run-name prefix -> (use_risk_signal, use_credit_score)
    "full": (True, True),
    "r_only": (True, False),
    "cs_only": (False, True),
    "none": (False, False),
}
DE_RISKING = {"deposit", "repay"}


def is_de_risking(action: int) -> bool:
    return action != HOLD_ACTION and ACTION_TYPES[action % len(ACTION_TYPES)] in DE_RISKING


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
    env = LendingEnv(seed=EVAL_SEED, split=split, use_risk_signal=use_r, use_credit_score=use_cs)
    rows = []
    for _ in range(episodes):
        obs, info = env.reset()
        high_risk = proactive = invalid = steps = 0
        ctrl = 0.0
        done = False
        while not done:
            a = policy(obs, info)
            obs, reward, terminated, truncated, info = env.step(a)
            steps += 1
            invalid += int(info["invalid_action"])
            ctrl += info["interest_usd"] + info["liquidation_loss_usd"]
            if info["risk_signal"] > SAFE_RISK_THRESHOLD:
                high_risk += 1
                proactive += int(is_de_risking(a) and not info["invalid_action"])
            done = terminated or truncated
        liq = bool(info["liquidated"])
        n_end = info["net_worth_usd"]
        rows.append({
            "strategy": name,
            "N_end": n_end,
            "liquidated": liq,
            "risk_adj": n_end / (1 + int(liq)),
            "proactive": proactive / high_risk if high_risk else np.nan,
            "ctrl_gain": ctrl,
            "invalid": invalid / steps,
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[], help="trained run names, e.g. full_seed42")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--episodes", type=int, default=200)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    probe = LendingEnv(seed=EVAL_SEED, split=args.split)
    n_obs, n_actions = probe.observation_space.shape[0], probe.action_space.n

    strategies = [
        ("static", lambda o, i: HOLD_ACTION, True, True),
        ("heuristic", lambda o, i: heuristic_policy(i), True, True),
        ("random", lambda o, i: int(rng.integers(0, n_actions)), True, True),
    ]
    for run in args.runs:
        use_r, use_cs = VARIANT_FLAGS[run.rsplit("_seed", 1)[0]]
        strategies.append((run, make_dqn_policy(run, n_obs, n_actions), use_r, use_cs))

    print(f"=== Evaluation on {args.split.upper()} split, {args.episodes} episodes each, "
          f"same price paths for every strategy ===\n")
    results = pd.concat([evaluate(n, pol, r, cs, args.split, args.episodes)
                         for n, pol, r, cs in strategies], ignore_index=True)

    order = [s[0] for s in strategies]
    table = (results.groupby("strategy", sort=False)
             .agg(N_end=("N_end", "mean"), liq_rate=("liquidated", "mean"),
                  risk_adj=("risk_adj", "mean"), proactive=("proactive", "mean"),
                  ctrl_gain=("ctrl_gain", "mean"), invalid=("invalid", "mean"))
             .reindex(order))
    pd.set_option("display.width", 120)
    print(table.round({"N_end": 1, "liq_rate": 3, "risk_adj": 1, "proactive": 3,
                       "ctrl_gain": 2, "invalid": 3}).to_string())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"eval_{args.split}.csv")
    results.to_csv(out, index=False)
    print(f"\nPer-episode results saved to {out}")


if __name__ == "__main__":
    main()