"""
evaluate.py  (v3.3 -- env v4.1: CS-based rates, new R, stress split, action counts, action fee)

Runs every strategy on the SAME price paths (same seed -> same paths and
same credit scores for everyone, because actions never use the env's random
generator), with no exploration, and prints three tables.

What changed in v3 (6 Nov, env v4):
  - int_paid uses each user's OWN borrow rate (it depends on CS in env v4).
  - NEW baseline heuristic_cs: the heuristic, but it KNOWS the CS rule
    (never borrows and pays debt down when the user's rate is above the 5%
    wallet yield). A fair, stronger hand-made rule to beat.
  - NEW split "stress": all paths synthetic, start stormy, 3x more crashes
    (python evaluate.py --split stress). Never used for training or tuning.
  - "high risk" now means R > 5% (chance of liquidation within 5 days),
    measured before acting, with the TRUE R also for no-R variants.
  - v3.1 (7 Nov): --seed (default 2026). With the same seed, VAL and TEST
    share the SAME users (CS) and the SAME synthetic price moves (only the
    real windows differ). Use a different seed for TEST (e.g. --seed 3031)
    so the final test is independent of VAL.
  - v3.2 (18 Nov): TABLE 3 -- action frequency. For every strategy, the share
    of steps each of the 9 actions was CHOSEN (pooled over all steps), plus
    the share that actually DID something (valid). Main use: check that a
    trained agent learned to avoid "borrow_ETH" (it costs interest and the
    borrowed ETH earns nothing in the wallet). Saved to
    results/actions_<split>_seed<seed>.csv
  - v3.3 (18 Nov): --fee X evaluates EVERY strategy (baselines too) in the
    world where each real action costs $X (env v4.1). ctrl_gain subtracts
    the fees; new column "fees" in TABLE 2. Output files get _feeX in the
    name, so the fee-free results are never overwritten.
    Compare fee-trained agents with the baselines in the SAME fee world.

Strategies:
  - static       : never rebalances
  - heuristic    : repays when H < 1.2, borrows when H > 2.0, else holds (CS-blind)
  - heuristic_cs : heuristic + the CS rule above (v3)
  - random       : random actions (sanity floor)
  - borrow_all   : always tries to borrow USDC; only the 1.05 rule stops it
Trained DQN runs are added with --runs, e.g. full_seed42_env4 none_seed42_env4.

TABLE 1 -- main results: N_end, liq_rate, liq_count, risk_adj, ctrl_gain, invalid
TABLE 2 -- behaviour (pooled over all steps): proactive, normal, pro_gap,
  high_risk, hf_lt_1.1, int_earned, int_paid, end_debt, idle_eth
TABLE 3 -- action frequency (% of steps each action was chosen)

Usage:
    python evaluate.py --runs full_seed42_env4 none_seed42_env4 --episodes 300
    python evaluate.py --runs ... --split stress --episodes 300
    python evaluate.py --runs ... --split test          # FINAL results only, run once
    python evaluate.py --runs cs_only_seed42_env4_fee0.05 --fee 0.05   # fee world
Per-episode results are saved to results/eval_<split>_seed<seed>.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))

from lending_env import (  # noqa: E402
    LendingEnv, ASSETS, ACTION_TYPES, HOLD_ACTION, SAFE_RISK_THRESHOLD,
    SUPPLY_APR, EXTERNAL_YIELD_APR, DAYS_PER_YEAR,
    heuristic_policy, heuristic_cs_policy, action_id, compute_risk_signal,
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
EVAL_SEED = 2026
LOW_HF = 1.1

VARIANT_FLAGS = {
    "full": (True, True),
    "r_only": (True, False),
    "cs_only": (False, True),
    "none": (False, False),
}
DE_RISKING = {"deposit", "repay"}


def action_name(action: int) -> str:
    """0 -> 'deposit_ETH', 6 -> 'borrow_USDC', 8 -> 'hold'."""
    if action == HOLD_ACTION:
        return "hold"
    return f"{ACTION_TYPES[action % len(ACTION_TYPES)]}_{ASSETS[action // len(ACTION_TYPES)]}"


ACTION_NAMES = [action_name(a) for a in range(HOLD_ACTION + 1)]


def is_de_risking(action: int) -> bool:
    return action != HOLD_ACTION and ACTION_TYPES[action % len(ACTION_TYPES)] in DE_RISKING


class EvalEnv(LendingEnv):
    """Same environment, but it also remembers the interest PAID and EARNED each day."""

    def _accrue_interest(self) -> float:
        p = {a: self._price(a) for a in ASSETS}
        self.last_paid = sum(self.debt[a] * p[a] * self.borrow_apr(a) for a in ASSETS) / DAYS_PER_YEAR
        self.last_earned = (self.wallet["USDC"] * EXTERNAL_YIELD_APR
                            + sum(self.collateral[a] * p[a] * SUPPLY_APR[a] for a in ASSETS)) / DAYS_PER_YEAR
        return super()._accrue_interest()


def risk_before_action(env: LendingEnv) -> tuple[float, float]:
    """(HF, R) of the state the strategy sees right now, before it acts."""
    _, _, hf, liq_prob, volatility_norm, utilization = env._risk_components()
    return hf, compute_risk_signal(liq_prob, volatility_norm, utilization)


def make_dqn_policy(run_name: str, n_obs: int, n_actions: int):
    import torch
    from dqn_agent import QNetwork, masked_argmax

    net = QNetwork(n_obs, n_actions)
    net.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, run_name, "final.pt"), map_location="cpu"))
    net.eval()

    def act(obs, info):
        with torch.no_grad():
            q = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        return masked_argmax(q, info["action_mask"])
    return act


def evaluate(name, policy, use_r, use_cs, split, episodes, seed=EVAL_SEED, fee=0.0):
    env = EvalEnv(seed=seed, split=split, use_risk_signal=use_r, use_credit_score=use_cs, action_fee=fee)
    rows = []
    for _ in range(episodes):
        obs, info = env.reset()
        c = dict(steps=0, invalid=0, high_steps=0, high_derisk=0, normal_steps=0, normal_derisk=0,
                 low_hf_steps=0)
        chosen = {n: 0 for n in ACTION_NAMES}          # v3.2: how often each action was picked
        valid = {n: 0 for n in ACTION_NAMES}           # ...and how often it actually did something
        ctrl = paid = earned = idle_eth = 0.0
        done = False
        while not done:
            hf_now, r_now = risk_before_action(env)
            idle_eth += env.wallet["ETH"] * env._price("ETH")
            a = int(policy(obs, info))
            obs, reward, terminated, truncated, info = env.step(a)

            chosen[action_name(a)] += 1
            valid[action_name(a)] += int(not info["invalid_action"])
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
            ctrl += info["interest_usd"] + info["liquidation_loss_usd"] - info["fee_usd"]
            paid += env.last_paid
            earned += env.last_earned
            done = terminated or truncated

        liq_count = int(info["liquidation_count"])
        n_end = info["net_worth_usd"]
        rows.append({
            "strategy": name,
            "credit_score": env.credit_score,
            "N_end": n_end,
            "liquidated": liq_count > 0,
            "liq_count": liq_count,
            "risk_adj": n_end / (1 + liq_count),
            "ctrl_gain": ctrl,
            "int_earned": earned,
            "int_paid": paid,
            "fees": info["fees_paid_usd"],
            "end_debt": info["debt_usd"],
            "idle_eth": idle_eth / c["steps"],
            **c,
            **{f"act_{n}": v for n, v in chosen.items()},
            **{f"ok_{n}": v for n, v in valid.items()},
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
    means = g[["int_earned", "int_paid", "fees", "end_debt", "idle_eth"]].mean().reindex(order)
    behaviour = behaviour.join(means)
    return main_table, behaviour


def action_table(results: pd.DataFrame, order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """v3.2: TABLE 3. Share of all steps each action was chosen (rows sum to 1),
    and the share where it was chosen AND did something (valid)."""
    g = results.groupby("strategy", sort=False)
    steps = g["steps"].sum().reindex(order)
    chosen = g[[f"act_{n}" for n in ACTION_NAMES]].sum().reindex(order)
    valid = g[[f"ok_{n}" for n in ACTION_NAMES]].sum().reindex(order)
    chosen.columns = valid.columns = ACTION_NAMES
    return chosen.div(steps, axis=0), valid.div(steps, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[], help="trained run names, e.g. full_seed42_env4")
    p.add_argument("--split", default="val", choices=["train", "val", "test", "stress"],
                   help="val (default) for tuning; stress for the stress test; test ONLY for final results")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=EVAL_SEED,
                   help="evaluation seed (users + price paths); use a different one for TEST, e.g. 3031")
    p.add_argument("--fee", type=float, default=0.0,
                   help="v3.3: $ fee per real action for EVERY strategy (env v4.1); 0 = off")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    probe = LendingEnv(seed=args.seed, split=args.split)
    n_obs, n_actions = probe.observation_space.shape[0], probe.action_space.n
    borrow_usdc = action_id("borrow", "USDC")

    strategies = [
        ("static", lambda o, i: HOLD_ACTION, True, True),
        ("heuristic", lambda o, i: heuristic_policy(i), True, True),
        ("heuristic_cs", lambda o, i: heuristic_cs_policy(i), True, True),
        ("random", lambda o, i: int(rng.integers(0, n_actions)), True, True),
        ("borrow_all", lambda o, i: borrow_usdc, True, True),
    ]
    for run in args.runs:
        use_r, use_cs = VARIANT_FLAGS[run.rsplit("_seed", 1)[0]]
        strategies.append((run, make_dqn_policy(run, n_obs, n_actions), use_r, use_cs))

    if args.split == "test":
        print("!!! TEST split: use this only for the FINAL results, not for tuning !!!\n")
    print(f"=== Evaluation on {args.split.upper()} split, {args.episodes} episodes each, seed {args.seed}, "
          f"fee ${args.fee:g} per action, same price paths for every strategy ===\n")
    results = pd.concat([evaluate(n, pol, r, cs, args.split, args.episodes, args.seed, args.fee)
                         for n, pol, r, cs in strategies], ignore_index=True)

    order = [s[0] for s in strategies]
    main_table, behaviour = summarise(results, order)
    chosen, valid = action_table(results, order)

    pd.set_option("display.width", 200)
    print("TABLE 1 - main results")
    print(main_table.round({"N_end": 1, "liq_rate": 3, "liq_count": 2, "risk_adj": 1,
                            "ctrl_gain": 2, "invalid": 3}).to_string())
    print(f"\nTABLE 2 - behaviour (high risk = R > {SAFE_RISK_THRESHOLD:.0%} chance of liquidation "
          f"within 5 days, measured before acting)")
    print(behaviour.round(3).round({"int_earned": 2, "int_paid": 2, "fees": 2, "end_debt": 0, "idle_eth": 0}).to_string())
    print("\nHow to read: pro_gap > 0 means it de-risks MORE when risk is high (real reaction to risk).")
    print("ctrl_gain = int_earned - int_paid - fees + liquidation loss.")

    print("\nTABLE 3 - action frequency (% of all steps the action was CHOSEN; each row sums to 100)")
    print((chosen * 100).round(1).to_string())
    print("\nTABLE 3b - % of all steps the action was chosen AND did something (valid)")
    print((valid * 100).round(1).to_string())
    print("\nHow to read: borrow_ETH should be ~0 for a trained agent (it pays interest, "
          "and borrowed ETH earns nothing in the wallet).")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"_fee{args.fee:g}" if args.fee > 0 else ""
    out = os.path.join(RESULTS_DIR, f"eval_{args.split}_seed{args.seed}{tag}.csv")
    results.to_csv(out, index=False)
    act_out = os.path.join(RESULTS_DIR, f"actions_{args.split}_seed{args.seed}{tag}.csv")
    pd.concat({"chosen": chosen, "valid": valid}, axis=1).to_csv(act_out)
    print(f"\nPer-episode results saved to {out}")
    print(f"Action frequency saved to {act_out}")


if __name__ == "__main__":
    main()