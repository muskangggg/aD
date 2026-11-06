"""
cs_analysis.py  (Step 1 of the upgrade plan -- read-only, no training)

QUESTION: does the Credit Score CS make the agent treat users DIFFERENTLY?
  If CS works as designed (losses weigh 1.5 - CS, gains 0.5 + CS), an agent
  that SEES CS should:
     low-CS user  -> play safe : higher HF, less debt
     high-CS user -> play bold : lower HF, more debt
  Agents that do NOT see CS (r_only, none) are the CONTROL GROUP: their
  behaviour should NOT change with CS (they can't see it).

  Example: a bank gives a careful student a small loan and a trusted
  customer a bigger one. We check whether the agent does the same.

WHAT IT DOES:
  Runs static, heuristic and your trained runs on the SAME price paths and
  the SAME users as evaluate.py (EVAL_SEED 2026), and records per episode:
     cs        : the user's credit score (real, also for agents that can't see it)
     avg_hf    : average HF over the 90 days (capped at 5, before each action)
     min_hf    : lowest HF in the episode
     avg_debt  : average debt in $ over the 90 days
     ctrl_gain : interest + liquidation loss in $ (same as evaluate.py)
     cs_score  : the CS-weighted reward = the goal a CS agent was TRAINED for
                 (ALPHA * m(CS) * money change - BETA per liquidation),
                 computed with the user's real CS for EVERY strategy so all
                 are judged on the same CS goal

PRINTS:
  TABLE A - behaviour per CS group (low / mid / high), mean over seeds
  TABLE B - the one-number test: rank correlation between CS and debt / HF
            per run, then mean +- std per group.
            CS working = debt goes UP with CS (corr > 0) and HF goes DOWN
            (corr < 0) for full / cs_only, and both close to 0 for the
            control groups r_only / none.
  TABLE C - money vs the CS goal, per group

Does NOT overwrite results/eval_<split>.csv. Saves results/cs_analysis_<split>.csv.

Usage:
    python cs_analysis.py --runs full_seed42_lr1e4 full_seed1_lr1e4 full_seed2_lr1e4 \
        r_only_seed42 r_only_seed1 r_only_seed2 cs_only_seed42 cs_only_seed1 cs_only_seed2 \
        none_seed42 none_seed1 none_seed2 --episodes 300
"""

import argparse
import os

import numpy as np
import pandas as pd

from evaluate import (
    EvalEnv, make_dqn_policy, risk_before_action, VARIANT_FLAGS, EVAL_SEED, RESULTS_DIR,
)
from lending_env import HOLD_ACTION, ALPHA, BETA, HF_CLIP, heuristic_policy

CS_LABELS = ["low", "mid", "high"]


def cs_multiplier(cs: float, money_change: float) -> float:
    """The proposal's credit multiplier, always with the user's REAL CS."""
    return 0.5 + cs if money_change >= 0 else 1.5 - cs


def run_strategy(name, group, policy, use_r, use_cs, split, episodes):
    env = EvalEnv(seed=EVAL_SEED, split=split, use_risk_signal=use_r, use_credit_score=use_cs)
    rows = []
    for ep in range(episodes):
        obs, info = env.reset()
        cs = float(env.credit_score)
        hfs, debts = [], []
        ctrl = cs_score = 0.0
        done = False
        while not done:
            hf_now, _ = risk_before_action(env)
            hfs.append(min(hf_now, HF_CLIP))
            debts.append(info["debt_usd"])
            a = policy(obs, info)
            obs, reward, terminated, truncated, info = env.step(a)
            money = info["interest_usd"] + info["liquidation_loss_usd"]
            ctrl += money
            cs_score += ALPHA * cs_multiplier(cs, money) * money - BETA * int(info["liquidated"])
            done = terminated or truncated
        rows.append({
            "strategy": name, "group": group, "episode": ep, "cs": cs,
            "avg_hf": float(np.mean(hfs)), "min_hf": float(np.min(hfs)),
            "avg_debt": float(np.mean(debts)), "end_debt": info["debt_usd"],
            "ctrl_gain": ctrl, "cs_score": cs_score,
            "liquidated": int(info["liquidation_count"]) > 0,
        })
    return pd.DataFrame(rows)


def rank_corr(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation (no scipy needed)."""
    return float(x.rank().corr(y.rank()))


def mean_std(values) -> str:
    v = np.asarray(values, dtype=float)
    if len(v) == 1:
        return f"{v[0]:+.2f}"
    return f"{v.mean():+.2f} +- {v.std(ddof=1):.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--episodes", type=int, default=300)
    args = p.parse_args()

    probe = EvalEnv(seed=EVAL_SEED, split=args.split)
    n_obs, n_actions = probe.observation_space.shape[0], probe.action_space.n

    strategies = [
        ("static", "static", lambda o, i: HOLD_ACTION, True, True),
        ("heuristic", "heuristic", lambda o, i: heuristic_policy(i), True, True),
    ]
    for run in args.runs:
        variant = run.rsplit("_seed", 1)[0]
        use_r, use_cs = VARIANT_FLAGS[variant]
        strategies.append((run, variant, make_dqn_policy(run, n_obs, n_actions), use_r, use_cs))

    print(f"=== CS analysis on {args.split.upper()} split, {args.episodes} episodes each, "
          f"same price paths and users as evaluate.py ===")
    df = pd.concat([run_strategy(n, g, pol, r, c, args.split, args.episodes)
                    for n, g, pol, r, c in strategies], ignore_index=True)

    # Same users for every strategy -> cut CS into 3 equal-size groups once
    users = df[df["strategy"] == "static"][["episode", "cs"]]
    cut = pd.qcut(users["cs"], 3, labels=CS_LABELS, retbins=True)
    edges = cut[1]
    df["cs_group"] = df["episode"].map(dict(zip(users["episode"], cut[0])))
    print(f"\nCS groups (equal size): low {edges[0]:.2f}-{edges[1]:.2f} | "
          f"mid {edges[1]:.2f}-{edges[2]:.2f} | high {edges[2]:.2f}-{edges[3]:.2f}")
    if df.groupby("strategy")["cs"].apply(lambda s: tuple(s.round(6))).nunique() != 1:
        print("WARNING: strategies did NOT see the same users -- comparison is not fair!")

    group_order = list(dict.fromkeys(df["group"]))
    pd.set_option("display.width", 160)

    # ---- TABLE A: behaviour per CS group (per run first, then mean over seeds) ----
    per_run = (df.groupby(["group", "strategy", "cs_group"], observed=True)
                 [["avg_hf", "min_hf", "avg_debt", "ctrl_gain", "liquidated"]].mean())
    table_a = per_run.groupby(["group", "cs_group"], observed=True).mean()
    table_a = table_a.reindex(pd.MultiIndex.from_product([group_order, CS_LABELS],
                                                         names=["group", "cs_group"]))
    table_a = table_a.rename(columns={"liquidated": "liq_rate"})
    print("\nTABLE A - behaviour per CS group (mean over seeds)")
    print("  CS working: for full / cs_only, avg_hf falls and avg_debt rises from low -> high CS.")
    print("  Control (r_only / none): rows should look about the same (they can't see CS).")
    print(table_a.round({"avg_hf": 3, "min_hf": 3, "avg_debt": 0, "ctrl_gain": 2, "liq_rate": 3}).to_string())

    # ---- TABLE B: the one-number test ----
    rows = []
    for (group, strat), d in df.groupby(["group", "strategy"], sort=False):
        lo, hi = d[d["cs_group"] == "low"], d[d["cs_group"] == "high"]
        rows.append({"group": group, "strategy": strat,
                     "corr_cs_debt": rank_corr(d["cs"], d["avg_debt"]),
                     "corr_cs_hf": rank_corr(d["cs"], d["avg_hf"]),
                     "debt_high_minus_low": hi["avg_debt"].mean() - lo["avg_debt"].mean(),
                     "hf_high_minus_low": hi["avg_hf"].mean() - lo["avg_hf"].mean()})
    per_run_b = pd.DataFrame(rows)
    print("\nTABLE B - per run: does behaviour follow CS?")
    print(per_run_b.round(3).to_string(index=False))
    print("\nTABLE B (summary) - mean +- std over seeds")
    summary_b = pd.DataFrame({
        col: per_run_b.groupby("group", sort=False)[col].apply(mean_std)
        for col in ["corr_cs_debt", "corr_cs_hf", "debt_high_minus_low", "hf_high_minus_low"]
    }).reindex(group_order)
    print(summary_b.to_string())
    print("  CS working: corr_cs_debt > 0 and corr_cs_hf < 0 for full / cs_only, "
          "clearly bigger than for r_only / none.")
    print("  Note: even the control can show a small correlation, because the same users always meet")
    print("  the same price paths. That is why we compare WITH-CS agents against the control, not against 0.")

    # ---- TABLE C: money vs the CS goal ----
    per_run_c = df.groupby(["group", "strategy"], sort=False)[["ctrl_gain", "cs_score"]].mean()
    table_c = pd.DataFrame({
        col: per_run_c.groupby("group", sort=False)[col].apply(mean_std) for col in ["ctrl_gain", "cs_score"]
    }).reindex(group_order)
    print("\nTABLE C - plain money (ctrl_gain, $) vs the CS goal the agent was trained for (cs_score)")
    print(table_c.to_string())
    print("  If full / cs_only rank better on cs_score than on ctrl_gain, they are doing what")
    print("  they were trained for: 'CS hurts money' is then a design choice, not a failure.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"cs_analysis_{args.split}.csv")
    df.to_csv(out, index=False)
    print(f"\nPer-episode results saved to {out}")


if __name__ == "__main__":
    main()