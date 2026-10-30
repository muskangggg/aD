"""
diagnose.py  -- WHY doesn't the agent learn to stay out of the danger zone?

Loads a trained model (no exploration, epsilon = 0) and runs 3 checks.
No training, no files changed -- it only looks.

CHECK 1 - Are the agent's value guesses right?
  For every day, compare what the network PREDICTS (Q of the action it took)
  with what REALLY happened afterwards (discounted sum of the real rewards,
  G = r0 + 0.95*r1 + 0.95^2*r2 + ...). Only days 1-50 are used, so at least
  40 future days are counted (0.95^40 = 0.13, the cut-off barely matters).
    - guesses close to reality (small gap, correlation > 0.5)
        -> the network learned the values; the problem is the reward/setup
    - guesses far off (big gap, low correlation)
        -> the LEARNING is broken (unstable / noisy), not the reward
  Split by HF zone, because the danger zone is where it matters.

CHECK 2 - What does it choose in each HF zone?
  Share of borrow / withdraw / de-risk (deposit, repay) / hold per zone.
  A careful agent borrows less and de-risks more as HF falls.

CHECK 3 - Does it SEE the danger? (HF sweep)
  Take one position and only change the debt so HF = 2.0 ... 1.05
  (everything else the same). Print the Q-values for hold / borrow USDC /
  repay USDC and the chosen action. A healthy agent's "repay" should rise
  above "borrow" as HF falls. If the Q-values barely change with HF,
  the network does not notice HF at all.

CHECK 0 (optional, --ckpts) - does the over-guess GROW during training?
  Runs only Check 1 (ALL row) for several saved checkpoints of one run.
    gap grows steadily      -> snowballing overestimation (learning diverges)
    gap appears early, flat -> a fixed bias, not a snowball

Usage:
    python diagnose.py --run full_seed42_buf1
    python diagnose.py --run full_seed42 --episodes 100
    python diagnose.py --run full_seed42 --ckpts ep100 ep300 ep500 ep1000 ep2000 final
The danger-zone penalty of the run is read from its name (_buf1 -> 1.0),
so the "real rewards" in Check 1 match what the agent was trained on.
"""

import argparse
import os
import re
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))

from lending_env import (  # noqa: E402
    LendingEnv, ACTION_TYPES, ASSETS, HOLD_ACTION, LIQUIDATION_THRESHOLD, action_id,
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DIAG_SEED = 2026          # same price paths as evaluate.py
CHECK_DAYS = 50           # days 1-50 used in Check 1
HF_ZONES = [(0.0, 1.1, "HF < 1.10"), (1.1, 1.3, "1.10-1.30"), (1.3, 1e9, "HF > 1.30")]
SWEEP_HFS = [2.0, 1.5, 1.3, 1.2, 1.15, 1.1, 1.05]

VARIANT_FLAGS = {"full": (True, True), "r_only": (True, False),
                 "cs_only": (False, True), "none": (False, False)}


def parse_run(run: str):
    """'full_seed42_buf1' -> (use_r, use_cs, buffer_penalty)."""
    use_r, use_cs = VARIANT_FLAGS[run.rsplit("_seed", 1)[0]]
    m = re.search(r"_buf([0-9.]+)$", run)
    return use_r, use_cs, float(m.group(1)) if m else 0.0


def load_q_function(run: str, n_obs: int, n_actions: int, ckpt: str = "final"):
    """Returns (q_fn, gamma). q_fn(obs) -> array of 9 Q-values."""
    import torch
    from dqn_agent import QNetwork, GAMMA

    net = QNetwork(n_obs, n_actions)
    net.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, run, f"{ckpt}.pt"), map_location="cpu"))
    net.eval()

    def q_fn(obs):
        with torch.no_grad():
            return net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return q_fn, GAMMA


def masked_best(q, mask):
    q = np.array(q, dtype=np.float64)
    q[~np.asarray(mask, dtype=bool)] = -np.inf
    return int(np.argmax(q))


def action_group(a: int) -> str:
    if a == HOLD_ACTION:
        return "hold"
    t = ACTION_TYPES[a % len(ACTION_TYPES)]
    return "de-risk" if t in ("deposit", "repay") else t


def action_name(a: int) -> str:
    if a == HOLD_ACTION:
        return "hold"
    return f"{ACTION_TYPES[a % len(ACTION_TYPES)]} {ASSETS[a // len(ACTION_TYPES)]}"


def zone_of(hf: float) -> str:
    for lo, hi, name in HF_ZONES:
        if lo <= hf < hi:
            return name
    return HF_ZONES[-1][2]


def collect(env, q_fn, gamma, episodes):
    """Greedy episodes. Per day: HF before acting, chosen action, predicted Q, reward."""
    days = []
    for _ in range(episodes):
        obs, info = env.reset()
        ep = []
        done = False
        while not done:
            hf_before = env._current_hf()
            q = q_fn(obs)
            a = masked_best(q, info["action_mask"])
            obs, reward, terminated, truncated, info = env.step(a)
            ep.append({"hf": hf_before, "a": a, "q_pred": float(q[a]), "r": reward})
            done = terminated or truncated
        g = 0.0                        # real discounted return, computed backwards
        for d in reversed(ep):
            g = d["r"] + gamma * g
            d["g_real"] = g
        liq_ep = info["liquidation_count"] > 0
        for i, d in enumerate(ep):
            d["day"] = i + 1
            d["liq_ep"] = liq_ep
        days.extend(ep)
    return days


def check1(days):
    print("CHECK 1 - value guesses vs reality (days 1-%d)" % CHECK_DAYS)
    print(f"  {'zone':<11} {'days':>6} {'predicted':>10} {'real':>9} {'avg |gap|':>10} {'corr':>6}")
    rows = [d for d in days if d["day"] <= CHECK_DAYS]
    for _, _, name in HF_ZONES + [(0, 0, "ALL")]:
        sel = rows if name == "ALL" else [d for d in rows if zone_of(d["hf"]) == name]
        if len(sel) < 5:
            print(f"  {name:<11} {len(sel):>6}   (too few days)")
            continue
        p = np.array([d["q_pred"] for d in sel])
        g = np.array([d["g_real"] for d in sel])
        corr = np.corrcoef(p, g)[0, 1] if p.std() > 1e-9 and g.std() > 1e-9 else float("nan")
        print(f"  {name:<11} {len(sel):>6} {p.mean():>10.2f} {g.mean():>9.2f} "
              f"{np.abs(p - g).mean():>10.2f} {corr:>6.2f}")
    print("  -> small gap + corr > 0.5: values learned OK.  Big gap or corr near 0: learning is broken.\n")


def check2(days):
    print("CHECK 2 - choices per HF zone")
    groups = ["borrow", "withdraw", "de-risk", "hold"]
    print(f"  {'zone':<11} {'days':>6} " + " ".join(f"{g:>9}" for g in groups))
    for _, _, name in HF_ZONES:
        sel = [d for d in days if zone_of(d["hf"]) == name]
        if not sel:
            print(f"  {name:<11} {0:>6}")
            continue
        shares = [np.mean([action_group(d["a"]) == g for d in sel]) for g in groups]
        print(f"  {name:<11} {len(sel):>6} " + " ".join(f"{s:>9.1%}" for s in shares))
    print("  -> careful agent: borrow goes DOWN and de-risk goes UP as HF falls.\n")


def check3(env, q_fn):
    print("CHECK 3 - HF sweep: same position, only the debt changes")
    env.reset()
    env.t = 30                               # day 30, so volatility is not 0
    coll_usd = env._usd(env.collateral)
    hold, borrow, repay = HOLD_ACTION, action_id("borrow", "USDC"), action_id("repay", "USDC")
    print(f"  {'HF':>5} {'Q hold':>9} {'Q borrow':>9} {'Q repay':>9} {'repay-borrow':>13}  chosen")
    first = last = None
    for hf in SWEEP_HFS:
        debt = coll_usd * LIQUIDATION_THRESHOLD / hf
        env.debt = {"ETH": 0.0, "USDC": debt}
        env.wallet = {"ETH": 0.0, "USDC": 400.0 + (debt - 400.0)}   # borrowed money sits in the wallet
        obs, mask = env._build_observation(), env.action_mask()
        q = q_fn(obs)
        diff = q[repay] - q[borrow]
        first = diff if first is None else first
        last = diff
        print(f"  {hf:>5.2f} {q[hold]:>9.2f} {q[borrow]:>9.2f} {q[repay]:>9.2f} {diff:>13.2f}  "
              f"{action_name(masked_best(q, mask))}")
    print(f"  -> 'repay-borrow' should GROW as HF falls (here: {first:+.2f} at HF 2.0 -> {last:+.2f} at HF 1.05).")
    print("     If it barely changes, the network does not notice HF.\n")


def check0(args, use_r, use_cs, buf):
    print(f"=== CHECK 0 - over-guess across training | {args.run} | {args.split.upper()} | "
          f"{args.episodes} episodes each ===\n")
    print(f"  {'checkpoint':<11} {'predicted':>10} {'real':>9} {'avg |gap|':>10} {'liq_rate':>9}")
    for ckpt in args.ckpts:
        env = LendingEnv(seed=DIAG_SEED, split=args.split, use_risk_signal=use_r,
                         use_credit_score=use_cs, buffer_penalty=buf)   # same price paths for each
        try:
            q_fn, gamma = load_q_function(args.run, env.observation_space.shape[0], env.action_space.n, ckpt)
        except FileNotFoundError:
            print(f"  {ckpt:<11} (file not found)")
            continue
        days = collect(env, q_fn, gamma, args.episodes)
        rows = [d for d in days if d["day"] <= CHECK_DAYS]
        p_ = np.array([d["q_pred"] for d in rows])
        g_ = np.array([d["g_real"] for d in rows])
        liq = np.mean([d["liq_ep"] for d in days if d["day"] == 1])
        print(f"  {ckpt:<11} {p_.mean():>10.2f} {g_.mean():>9.2f} {np.abs(p_ - g_).mean():>10.2f} {liq:>9.3f}")
    print("\n  -> gap grows steadily: snowballing overestimation.  Early and flat: a fixed bias.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="full_seed42")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--ckpts", nargs="*", default=None,
                   help="checkpoint names, e.g. ep100 ep500 final -> runs Check 0 only")
    args = p.parse_args()

    use_r, use_cs, buf = parse_run(args.run)
    if args.ckpts:
        check0(args, use_r, use_cs, buf)
        return
    env = LendingEnv(seed=DIAG_SEED, split=args.split, use_risk_signal=use_r,
                     use_credit_score=use_cs, buffer_penalty=buf)
    q_fn, gamma = load_q_function(args.run, env.observation_space.shape[0], env.action_space.n)

    print(f"=== diagnose {args.run} | {args.split.upper()} | {args.episodes} episodes | "
          f"buffer_penalty={buf:g} | epsilon=0 ===\n")
    days = collect(env, q_fn, gamma, args.episodes)
    check1(days)
    check2(days)
    check3(env, q_fn)


if __name__ == "__main__":
    main()