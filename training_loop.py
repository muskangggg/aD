"""
training_loop.py  (v4.1)

Step 8: the training loop, on top of Step 7's DQNAgent.

What changed in v4.1 (18 Nov, env v4.1):
  - --fee X turns on the env's action fee ($X per action that does something,
    like gas; default 0 = off = exactly v4). The run name gets _feeX at the
    end, e.g. cs_only_seed42_env4_fee0.05, so the v4 runs are never touched.
  - controllable_gain now subtracts the fee (it is real money). With
    --fee 0 the number is the same as before.

What changed in v4 (6 Nov, env v4):
  - INFO-ONLY ABLATION: the replay boost (PER x3 for high-risk moments) now
    uses the TRUE risk R for EVERY variant. Before v4 the "no R" variants
    also lost the boost, so they differed in TWO ways (what they see AND how
    they replay). Now --no-r / --no-cs only hide information, so a
    difference between variants means "seeing it helps".
  - RUN NAME TAG: --tag (default "env4") is added at the end of the run name,
    e.g. full_seed42_env4, so the new runs never overwrite the old ones
    (r_only_seed42, none_seed1, ...). evaluate.py / diagnose.py /
    inspect_policy.py read these names without changes.
  - the log column danger_days_share stays (HF < 1.15); new column
    high_risk_share = share of days with R > 5% (before acting).

What changed in v3.1 / v3 (kept):
  - --buffer-penalty X turns on the env's danger-zone penalty (off by default).
  - TIME-LIMIT FIX (Pardo et al., 2018): only `terminated` stops the future
    value in the learning target; day 90 (truncated) is just where we stop watching.
  - "proactive" measures risk BEFORE the action.

Variants:
    python training_loop.py                    -> full   (R + CS visible)
    python training_loop.py --no-cs            -> r_only
    python training_loop.py --no-r             -> cs_only
    python training_loop.py --no-r --no-cs     -> none
    add --seed N (default 42), --episodes N, --tag NAME, --fee X

Usage:
    python training_loop.py [--no-r] [--no-cs] [--seed N] [--episodes N] [--tag env4] [--fee 0.05]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))

from lending_env import (  # noqa: E402
    LendingEnv, ACTION_TYPES, HOLD_ACTION, SAFE_RISK_THRESHOLD, HF_BUFFER, compute_risk_signal,
)
from dqn_agent import DQNAgent  # noqa: E402

N_EPISODES = 1000
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CHECKPOINT_EVERY = 100
LOG_PRINT_EVERY = 20

DE_RISKING_ACTION_TYPES = {"deposit", "repay"}


def variant_name(use_r: bool, use_cs: bool) -> str:
    if use_r and use_cs:
        return "full"
    if use_r:
        return "r_only"
    if use_cs:
        return "cs_only"
    return "none"


def is_de_risking_action(action: int) -> bool:
    if action == HOLD_ACTION:
        return False
    return ACTION_TYPES[action % len(ACTION_TYPES)] in DE_RISKING_ACTION_TYPES


def risk_before_action(env: LendingEnv) -> float:
    """True R of the state the agent sees right now (also for no-R variants)."""
    _, _, _, liq_prob, volatility_norm, utilization = env._risk_components()
    return compute_risk_signal(liq_prob, volatility_norm, utilization)


def run_episode(env: LendingEnv, agent: DQNAgent, train: bool = True) -> dict:
    obs, info = env.reset()
    start_net_worth = info["net_worth_usd"]

    episode_rewards = []
    elevated_risk_steps = 0
    proactive_steps = 0
    invalid_steps = 0
    danger_days = 0
    action_steps = 0          # v4.1: steps where an action (not hold) really happened
    controllable_gain = 0.0
    liq_count = 0
    end_net_worth = start_net_worth

    done = False
    while not done:
        r_before = risk_before_action(env)
        action = agent.select_action(obs, info["action_mask"])
        next_obs, reward, terminated, truncated, info = env.step(action)

        if train:
            # v4: the replay boost uses the TRUE R for every variant (info-only ablation)
            agent.store_transition(obs, action, reward, next_obs, terminated,
                                   info["risk_signal"], info["action_mask"])
            agent.learn()

        episode_rewards.append(reward)
        invalid_steps += int(info["invalid_action"])
        action_steps += int(action != HOLD_ACTION and not info["invalid_action"])
        danger_days += int(info["health_factor"] < HF_BUFFER)
        controllable_gain += info["interest_usd"] + info["liquidation_loss_usd"] - info.get("fee_usd", 0.0)
        if r_before > SAFE_RISK_THRESHOLD:
            elevated_risk_steps += 1
            if is_de_risking_action(action) and not info["invalid_action"]:
                proactive_steps += 1

        liq_count = info["liquidation_count"]
        end_net_worth = info["net_worth_usd"]

        obs = next_obs
        done = terminated or truncated

    n = len(episode_rewards)
    return {
        "total_reward": float(np.sum(episode_rewards)),
        "episode_length": n,
        "liquidated": liq_count > 0,
        "liquidation_count": liq_count,
        "net_worth_end": end_net_worth,
        "net_wealth_change": end_net_worth - start_net_worth,
        "controllable_gain": controllable_gain,
        "risk_adjusted_return": end_net_worth / (1 + liq_count),
        "proactive_response_rate": (proactive_steps / elevated_risk_steps) if elevated_risk_steps > 0 else None,
        "invalid_action_rate": invalid_steps / n,
        "action_rate": action_steps / n,
        "fees_paid": info.get("fees_paid_usd", 0.0),
        "danger_days_share": danger_days / n,
        "high_risk_share": elevated_risk_steps / n,
        "credit_score": env.credit_score,
        "epsilon": agent.epsilon,
        "target_net_active": agent.target_net_active,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train the DQN agent.")
    p.add_argument("--no-r", action="store_true", help="hide the Risk Signal R (and vol, utilization)")
    p.add_argument("--no-cs", action="store_true", help="hide the Credit Score CS")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--tag", default="env4", help="added to the run name (default env4)")
    p.add_argument("--buffer-penalty", type=float, default=0.0,
                   help="HF danger-zone penalty per day at HF 1.05 (0 = off)")
    p.add_argument("--fee", type=float, default=0.0,
                   help="v4.1: $ fee per action that does something, like gas (0 = off)")
    return p.parse_args()


def main():
    args = parse_args()
    use_r, use_cs = not args.no_r, not args.no_cs
    run_name = f"{variant_name(use_r, use_cs)}_seed{args.seed}"
    if args.tag:
        run_name += f"_{args.tag}"
    if args.buffer_penalty > 0:
        run_name += f"_buf{args.buffer_penalty:g}"
    if args.fee > 0:
        run_name += f"_fee{args.fee:g}"

    run_ckpt_dir = os.path.join(CHECKPOINT_DIR, run_name)
    if os.path.exists(os.path.join(run_ckpt_dir, "final.pt")):
        raise SystemExit(f"STOP: {run_ckpt_dir} already has final.pt -- rename it or use another --tag.")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(run_ckpt_dir, exist_ok=True)

    print(f"=== Training run: {run_name} (R={'on' if use_r else 'off'}, "
          f"CS={'on' if use_cs else 'off'}, buffer_penalty={args.buffer_penalty:g}, "
          f"fee=${args.fee:g}, episodes={args.episodes}, split=train) ===")

    env = LendingEnv(seed=args.seed, use_risk_signal=use_r, use_credit_score=use_cs, split="train",
                     buffer_penalty=args.buffer_penalty, action_fee=args.fee)
    agent = DQNAgent(obs_dim=env.observation_space.shape[0], n_actions=env.action_space.n, seed=args.seed)

    log_rows = []
    for ep in range(1, args.episodes + 1):
        metrics = run_episode(env, agent, train=True)
        metrics["episode"] = ep
        log_rows.append(metrics)

        if ep % LOG_PRINT_EVERY == 0:
            recent = log_rows[-LOG_PRINT_EVERY:]
            proactive_vals = [r["proactive_response_rate"] for r in recent if r["proactive_response_rate"] is not None]
            print(f"ep {ep:4d} | eps={agent.epsilon:.3f} | target={'on ' if agent.target_net_active else 'off'} | "
                  f"avg_reward={np.mean([r['total_reward'] for r in recent]):8.3f} | "
                  f"liq_rate={np.mean([r['liquidated'] for r in recent]):.2f} | "
                  f"ctrl_gain={np.mean([r['controllable_gain'] for r in recent]):6.2f} | "
                  f"proactive={np.mean(proactive_vals) if proactive_vals else float('nan'):.2f} | "
                  f"act_rate={np.mean([r['action_rate'] for r in recent]):.2f} | "
                  f"high_risk={np.mean([r['high_risk_share'] for r in recent]):.2f} | "
                  f"invalid={np.mean([r['invalid_action_rate'] for r in recent]):.2f}")

        if ep % CHECKPOINT_EVERY == 0:
            torch.save(agent.online_net.state_dict(), os.path.join(run_ckpt_dir, f"ep{ep}.pt"))

    final_ckpt = os.path.join(run_ckpt_dir, "final.pt")
    torch.save(agent.online_net.state_dict(), final_ckpt)

    df = pd.DataFrame(log_rows)
    log_path = os.path.join(LOG_DIR, f"{run_name}.csv")
    df.to_csv(log_path, index=False)

    last = df.tail(100)
    print(f"\nSaved log: {log_path}")
    print(f"Final checkpoint: {final_ckpt}")
    print(f"\nLast 100 episodes (TRAIN):")
    print(f"  liquidation rate:      {last['liquidated'].mean():.3f}")
    print(f"  avg reward:            {last['total_reward'].mean():.3f}")
    print(f"  avg net wealth change: {last['net_wealth_change'].mean():.2f}")
    print(f"  avg controllable gain: {last['controllable_gain'].mean():.2f}")
    print(f"  avg risk-adj. return:  {last['risk_adjusted_return'].mean():.2f}")
    print(f"  avg proactive rate:    {last['proactive_response_rate'].dropna().mean():.3f}")
    print(f"  avg action rate:       {last['action_rate'].mean():.3f}  (share of days with a real action)")
    print(f"  avg fees paid:         {last['fees_paid'].mean():.2f}")
    print(f"  avg invalid rate:      {last['invalid_action_rate'].mean():.3f}")
    print(f"  high-risk days (R>{SAFE_RISK_THRESHOLD:.0%}): {last['high_risk_share'].mean():.3f}")
    print(f"  danger days (HF<{HF_BUFFER}):  {last['danger_days_share'].mean():.3f}")


if __name__ == "__main__":
    main()