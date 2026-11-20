"""
diagnose_r.py  (v1.1, 19 Nov -- improvement plan item 3; read-only, NO training)

QUESTION: R (chance of liquidation within 5 days) makes agents REACT more
(higher pro_gap), but in STRESS the R agents were liquidated MORE
(r_only 0.100 vs none 0.046; full 0.071 vs cs_only 0.050). Why?

For every liquidation it looks at the 5 days BEFORE it and sorts it into:
  crash_no_warning : R was <= 5% every day before, and the liquidation day
                     had a one-day ETH drop of 10% or more (a jump).
                     -> nobody could see it coming; R cannot help here.
  quiet_slide      : R was <= 5% every day before, no big one-day drop.
                     -> R missed a slow danger (R itself may be too weak).
  warned_ignored   : R was > 5% on at least one of the 5 days before, and
                     the agent made NO de-risking action (repay / deposit).
                     -> the agent saw the warning and did nothing.
  warned_too_slow  : R warned and the agent DID de-risk, but still got
                     liquidated. -> it reacted, but not enough / not fast enough.
                     For these (and warned_ignored) it also computes how
                     many $100 repays would have been needed on the first
                     warning day to push R below 5%, vs how many warning
                     days there were. needed > days -> "$100 per day is too slow".

v1.1 (19 Nov): TABLE 2b -- did the agent UNDO its safety moves? For every
warned liquidation it counts, inside the warning window (first day R > 5%
up to the liquidation day):
     derisk : repay / deposit that did something   (makes the loan safer)
     rerisk : borrow / withdraw that did something (makes the loan riskier)
     undid  : at least one derisk AND at least one rerisk in the window
     hf_first -> hf_last : HF on the first warning day and on the last
                           decision before the liquidation
     debt_change : debt on the last decision minus debt on the first warning day
  If "undid" is high and debt_change >= 0, the agent put the fire out with a
  cup of water and then poured petrol back on (churn near the edge).

It also checks "do R agents sit closer to the edge?":
  average HF and R on STORMY vs CALM days, and how often each agent
  de-risks on stormy days.

Example: a smoke alarm (R). Did the fire start with an explosion (crash, no
alarm possible)? Did the alarm ring and you stayed on the sofa (ignored)?
Or did you run for water but only had a cup (too slow)?

Usage (defaults: STRESS split, seed 2026, 300 episodes, the 20 env4 runs):
    python diagnose_r.py | tee results/diagnose_r_stress.txt
    python diagnose_r.py --runs full_seed42_env4 r_only_seed42_env4 --episodes 100
    python diagnose_r.py --split val
Saves results/diagnose_r_<split>_seed<seed>_events.csv (one row per liquidation)
and results/diagnose_r_<split>_seed<seed>_days.csv (stormy/calm summary per run).
"""

import argparse
import math
import os
import sys
from collections import deque

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))

from lending_env import (  # noqa: E402
    LendingEnv, ACTION_TYPES, HOLD_ACTION, SAFE_RISK_THRESHOLD, LIQUIDATION_THRESHOLD,
    ACTION_SIZE_USD, heuristic_policy, heuristic_cs_policy,
)
from compute_risk_signal import RISK_HORIZON_DAYS  # noqa: E402
from evaluate import make_dqn_policy, VARIANT_FLAGS, RESULTS_DIR, EVAL_SEED  # noqa: E402

LOOKBACK_DAYS = 5          # look at the 5 days before each liquidation (= R's horizon)
CRASH_DROP = 0.10          # a one-day ETH fall of 10% or more counts as a crash jump
Z_5PCT = 1.959964          # R = 2*Phi(-z) = 5%  <=>  z = 1.96
DEFAULT_RUNS = [f"{v}_seed{s}_env4" for v in ("full", "r_only", "cs_only", "none") for s in (42, 1, 2, 3, 4)]
CAUSES = ["crash_no_warning", "quiet_slide", "warned_ignored", "warned_too_slow"]


def is_de_risking(action: int) -> bool:
    return action != HOLD_ACTION and ACTION_TYPES[action % len(ACTION_TYPES)] in ("deposit", "repay")


def is_re_risking(action: int) -> bool:
    return action != HOLD_ACTION and ACTION_TYPES[action % len(ACTION_TYPES)] in ("borrow", "withdraw")


def true_r(env: LendingEnv) -> float:
    """True R of the current state (also for agents that cannot see R)."""
    return env._liquidation_probability()


def repays_needed_for_safe_r(env: LendingEnv) -> float:
    """How many $100 USDC repays would bring R below 5% right now (current
    volatility), for the usual case: ETH collateral, debt mostly USDC.
    R < 5%  <=>  log distance d > 1.96 * sigma * sqrt(5)
            <=>  USDC debt <= LT*coll_USDC + (LT*coll_ETH - debt_ETH) * exp(-d*)
    Returns nan if the position is not of that type."""
    p = env._price("ETH")
    a = LIQUIDATION_THRESHOLD * env.collateral["ETH"] * p - env.debt["ETH"] * p
    if a <= 0:
        return float("nan")
    d_star = Z_5PCT * env.sigma_daily() * math.sqrt(RISK_HORIZON_DAYS)
    max_usdc_debt = LIQUIDATION_THRESHOLD * env.collateral["USDC"] + a * math.exp(-d_star)
    excess = env.debt["USDC"] - max_usdc_debt
    return max(0.0, excess) / ACTION_SIZE_USD


def run_strategy(name, group, policy, use_r, use_cs, split, episodes, seed):
    env = LendingEnv(seed=seed, split=split, use_risk_signal=use_r, use_credit_score=use_cs)
    events, days = [], []
    for ep in range(episodes):
        obs, info = env.reset()
        regimes = env.price_sim.last_regimes          # 1 = stormy day (synthetic paths), None = real path
        history = deque(maxlen=LOOKBACK_DAYS)         # pre-action snapshots of the last 5 days
        n_liq = 0
        done = False
        while not done:
            d = env.t
            r_now, hf_now = true_r(env), env._current_hf()
            snap = {"day": d, "R": r_now, "hf": hf_now, "repays_needed": repays_needed_for_safe_r(env),
                    "wallet_usdc": env.wallet["USDC"], "debt": env._usd(env.debt)}
            a = int(policy(obs, info))
            obs, reward, terminated, truncated, info = env.step(a)
            snap["derisk"] = is_de_risking(a) and not info["invalid_action"]
            snap["rerisk"] = is_re_risking(a) and not info["invalid_action"]
            stormy = None if regimes is None else int(regimes[d])
            days.append({"strategy": name, "group": group, "stormy": stormy, "hf": min(hf_now, 5.0),
                         "R": r_now, "high_R": r_now > SAFE_RISK_THRESHOLD, "derisk": snap["derisk"]})

            if info["liquidated"]:
                window = list(history) + [snap]        # the 5 days before + the liquidation day's decision
                warned = [s for s in window if s["R"] > SAFE_RISK_THRESHOLD]
                day_drop = 1.0 - env.price_path[d + 1] / env.price_path[d]
                n_derisk = sum(s["derisk"] for s in window)
                if not warned:
                    cause = "crash_no_warning" if day_drop >= CRASH_DROP else "quiet_slide"
                else:
                    cause = "warned_too_slow" if n_derisk > 0 else "warned_ignored"
                first = warned[0] if warned else None
                w = window[window.index(first):] if first else []     # first warning day -> liquidation day
                last = window[-1]
                n_d = sum(x["derisk"] for x in w)
                n_r = sum(x["rerisk"] for x in w)
                events.append({
                    "strategy": name, "group": group, "episode": ep, "day": d, "cause": cause,
                    "first_in_episode": n_liq == 0,
                    "day_drop": day_drop, "stormy": stormy,
                    "max_R_before": max(s["R"] for s in window),
                    "warning_days": len(window) - window.index(first) if first else 0,
                    "derisk_actions": n_derisk,
                    "repays_needed_at_first_warning": first["repays_needed"] if first else np.nan,
                    "wallet_usdc_at_first_warning": first["wallet_usdc"] if first else np.nan,
                    "derisk_in_warning": n_d,
                    "rerisk_in_warning": n_r,
                    "undid": bool(first) and n_d > 0 and n_r > 0,
                    "hf_first_warning": first["hf"] if first else np.nan,
                    "hf_last": last["hf"],
                    "debt_change": last["debt"] - first["debt"] if first else np.nan,
                })
                n_liq += 1
            history.append(snap)
            done = terminated or truncated
    return pd.DataFrame(events), pd.DataFrame(days)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    p.add_argument("--split", default="stress", choices=["train", "val", "test", "stress"])
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    args = p.parse_args()

    probe = LendingEnv(seed=args.seed, split=args.split)
    n_obs, n_actions = probe.observation_space.shape[0], probe.action_space.n
    strategies = [
        ("static", "static", lambda o, i: HOLD_ACTION, True, True),
        ("heuristic", "heuristic", lambda o, i: heuristic_policy(i), True, True),
        ("heuristic_cs", "heuristic_cs", lambda o, i: heuristic_cs_policy(i), True, True),
    ]
    for run in args.runs:
        group = run.rsplit("_seed", 1)[0]
        use_r, use_cs = VARIANT_FLAGS[group]
        strategies.append((run, group, make_dqn_policy(run, n_obs, n_actions), use_r, use_cs))

    print(f"=== R diagnosis on {args.split.upper()} split, {args.episodes} episodes each, seed {args.seed}, "
          f"same price paths for every strategy (no fee) ===\n")
    ev_parts, day_parts = [], []
    for s in strategies:
        e, d = run_strategy(*s, args.split, args.episodes, args.seed)
        ev_parts.append(e)
        day_parts.append(d)
        print(f"  done: {s[0]:24s} liquidations = {len(e)}")
    events = pd.concat(ev_parts, ignore_index=True)
    days = pd.concat(day_parts, ignore_index=True)
    if events.empty:
        print("\nNo liquidations at all -- nothing to diagnose.")
        return
    group_order = list(dict.fromkeys(s[1] for s in strategies))
    n_runs = pd.Series({g: sum(1 for s in strategies if s[1] == g) for g in group_order})
    pd.set_option("display.width", 200)

    # ---- TABLE 1: why did the liquidations happen? ----
    # Only the FIRST liquidation of each episode is classified (= what liq_rate counts);
    # later ones in the same episode are usually the same bad position again.
    all_per_run = events.groupby("group").size().reindex(group_order, fill_value=0) / n_runs
    first = events[events["first_in_episode"]]
    counts = (first.groupby(["group", "cause"]).size().unstack(fill_value=0)
              .reindex(index=group_order, columns=CAUSES, fill_value=0))
    table1 = counts.div(n_runs, axis=0)                        # per run (= per --episodes episodes)
    table1.insert(0, "episodes_liq", table1.sum(axis=1))
    table1.insert(1, "all_liqs", all_per_run)
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    print(f"\nTABLE 1 - FIRST liquidation of each episode, by cause, per run ({args.episodes} episodes, "
          f"mean over seeds); all_liqs = every liquidation incl. repeats")
    print(table1.round(1).to_string())
    print("\nTABLE 1b - share of each group's liquidations by cause (%)")
    print(shares.round(0).to_string())
    print("  crash_no_warning: R could not help | quiet_slide: R missed it | "
          "warned_ignored: agent did nothing | warned_too_slow: reacted, not enough")

    # ---- TABLE 2: warned liquidations -- was $100 per day enough? ----
    warned = first[first["cause"].isin(["warned_ignored", "warned_too_slow"])]
    if len(warned):
        t2 = warned.groupby("group").agg(
            n=("cause", "size"),
            warning_days=("warning_days", "median"),
            derisk_actions=("derisk_actions", "median"),
            repays_needed=("repays_needed_at_first_warning", "median"),
            wallet_usdc=("wallet_usdc_at_first_warning", "median"),
        ).reindex([g for g in group_order if g in warned["group"].unique()])
        too_slow = warned.assign(slow=warned["repays_needed_at_first_warning"] > warned["warning_days"])
        t2["needed>days_%"] = too_slow.groupby("group")["slow"].mean() * 100
        print("\nTABLE 2 - warned liquidations (medians): warning days vs $100 repays needed to get R < 5%")
        print(t2.round(1).to_string())
        print("  needed > warning days -> even a perfect agent could not get safe with $100 per day.")
        print("  wallet_usdc < repays_needed x 100 -> not even enough cash in the wallet to repay.")

        # ---- TABLE 2b (v1.1): did the agent undo its safety moves? ----
        t2b = warned.groupby("group").agg(
            n=("cause", "size"),
            derisk=("derisk_in_warning", "mean"),
            rerisk=("rerisk_in_warning", "mean"),
            any_rerisk_pct=("rerisk_in_warning", lambda x: (x > 0).mean() * 100),
            undid_pct=("undid", lambda x: x.mean() * 100),
            hf_first=("hf_first_warning", "median"),
            hf_last=("hf_last", "median"),
            debt_change=("debt_change", "median"),
        ).reindex(t2.index)
        print("\nTABLE 2b - inside the warning window (first R > 5% day -> liquidation): "
              "safety moves vs risky moves")
        print(t2b.round(2).to_string())
        print("  derisk = repay/deposit, rerisk = borrow/withdraw (means per liquidation); "
              "undid_pct = % with BOTH; debt_change in $ (median).")
        print("  Churn confirmed if undid_pct / any_rerisk_pct are high for R agents and debt_change >= 0.")

    # ---- TABLE 3: do R agents sit closer to the edge? ----
    known = days.dropna(subset=["stormy"])
    if len(known):
        t3 = known.groupby(["group", "stormy"]).agg(
            avg_hf=("hf", "mean"), avg_R=("R", "mean"),
            high_R_days=("high_R", "mean"), derisk_rate=("derisk", "mean"),
        ).unstack("stormy")
        t3.columns = [f"{m}_{'storm' if s == 1 else 'calm'}" for m, s in t3.columns]
        t3 = t3.reindex(group_order)
        print("\nTABLE 3 - calm vs stormy days (synthetic paths only), mean over all days")
        print(t3.round(3).to_string())
        print("  R agents closer to the edge = lower avg_hf_storm / higher high_R_days_storm than cs_only / none.")
    else:
        print("\nTABLE 3 skipped: no synthetic paths in this split (use --split stress).")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    base = os.path.join(RESULTS_DIR, f"diagnose_r_{args.split}_seed{args.seed}")
    events.to_csv(base + "_events.csv", index=False)
    if len(known):
        t3.to_csv(base + "_days.csv")
    print(f"\nSaved {base}_events.csv (one row per liquidation)")


if __name__ == "__main__":
    main()