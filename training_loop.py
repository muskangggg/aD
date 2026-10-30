"""
training_loop.py  (v3.1)

Step 8: the training loop, on top of Step 7's DQNAgent.

What changed in v3.1:
  - --buffer-penalty X turns on the env's HF danger-zone penalty (lending_env
    v3.3, Step 2b experiment). The run name gets "_buf<X>" at the end, e.g.
    full_seed42_buf1, so it never overwrites runs without the penalty.
    evaluate.py / inspect_policy.py read these names without changes.
  - New log column danger_days_share: share of days with HF < HF_BUFFER (1.15).

What changed in v3:
  - TIME-LIMIT FIX (Pardo et al., 2018, "Time Limits in Reinforcement
    Learning"): day 90 is only where we STOP WATCHING (truncated), not a
    real end of the loan. Before v3 the agent was told "no future after
    day 90". The agent cannot see the day number, so the SAME situation got
    "the future matters" on most days and "no future" on day 90 --
    contradictory teaching. Now only `terminated` (collateral wiped out)
    stops the future value in the learning target. The episode still ends
    at day 90 as before.
  - "proactive" in the log now measures risk BEFORE the action (the state
    the agent saw when it chose), the same way evaluate.py v2 does.

What changed in v2:
  - Trains on the TRAIN price split only -- val/test are kept for evaluation.
  - Command-line options choose the model variant and seed, so the same
    file trains all 4 ablation models:
        python training_loop.py                    -> full   (R + CS)
        python training_loop.py --no-cs            -> r_only
        python training_loop.py --no-r             -> cs_only
        python training_loop.py --no-r --no-cs     -> none
        add --seed N for a different random seed (default 42)
  - Each run saves to its own files, e.g.
        logs/full_seed42.csv
        checkpoints/full_seed42/ep100.pt ... final.pt
  - Logs invalid_action_rate. The "no R" variant also loses R's replay
    boost (PER), so the ablation is clean.

Per-episode metrics (the proposal's 4, plus helpers):
  - net_worth_end           : N_user at episode end (primary metric)
  - net_wealth_change       : N_user end - start (includes the ETH price move)
  - controllable_gain       : interest + liquidation loss (the part actions control)
  - liquidated              : at least one liquidation in the episode
  - risk_adjusted_return    : N_user / (1 + liquidation_count)   (proposal Sec 5.4)
  - proactive_response_rate : of the steps where R (before acting) was
                              > SAFE_RISK_THRESHOLD, the fraction where the
                              agent made a deposit or repay THAT ACTUALLY WORKED

Usage:
    python training_loop.py [--no-r] [--no-cs] [--seed N] [--episodes N]
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

# ---- Training config (not specified by the proposal -- tune as needed) ----
N_EPISODES = 500          # ~45,000 steps at 90 steps/episode
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CHECKPOINT_EVERY = 100     # episodes
LOG_PRINT_EVERY = 20       # episodes

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
    """True if this action is a deposit or repay (on any asset)."""
    if action == HOLD_ACTION:
        return False
    return ACTION_TYPES[action % len(ACTION_TYPES)] in DE_RISKING_ACTION_TYPES


def risk_before_action(env: LendingEnv) -> float:
    """True R of the state the agent sees right now (also for no-R variants)."""
    _, _, _, health_deficit, volatility_norm, utilization = env._risk_components()
    return compute_risk_signal(health_deficit, volatility_norm, utilization)


def run_episode(env: LendingEnv, agent: DQNAgent, train: bool = True) -> dict:
    """Run one episode. train=True: explore + learn. train=False: the agent's
    current epsilon is used as-is and nothing is learned."""
    obs, info = env.reset()
    start_net_worth = info["net_worth_usd"]

    episode_rewards = []
    elevated_risk_steps = 0
    proactive_steps = 0
    invalid_steps = 0
    danger_days = 0
    controllable_gain = 0.0
    liq_count = 0
    end_net_worth = start_net_worth

    done = False
    while not done:
        r_before = risk_before_action(env)
        action = agent.select_action(obs, info["action_mask"])   # only actions that can work
        next_obs, reward, terminated, truncated, info = env.step(action)

        if train:
            # the "no R" ablation must not get R's replay boost either
            per_risk = info["risk_signal"] if env.use_risk_signal else 0.0
            # v3: only a TRUE end (terminated) stops the future value.
            # Day 90 (truncated) is just where we stop watching.
            agent.store_transition(obs, action, reward, next_obs, terminated,
                                   per_risk, info["action_mask"])
            agent.learn()

        episode_rewards.append(reward)
        invalid_steps += int(info["invalid_action"])
        danger_days += int(info["health_factor"] < HF_BUFFER)
        controllable_gain += info["interest_usd"] + info["liquidation_loss_usd"]
        if r_before > SAFE_RISK_THRESHOLD:
            elevated_risk_steps += 1
            # only count de-risking actions that actually did something
            if is_de_risking_action(action) and not info["invalid_action"]:
                proactive_steps += 1

        liq_count = info["liquidation_count"]   # liquidations so far (episode no longer ends at one)
        end_net_worth = info["net_worth_usd"]

        obs = next_obs
        done = terminated or truncated

    total_reward = float(np.sum(episode_rewards))

    return {
        "total_reward": total_reward,
        "episode_length": len(episode_rewards),
        "liquidated": liq_count > 0,                           # at least one liquidation
        "liquidation_count": liq_count,
        "net_worth_end": end_net_worth,                        # N_user (primary metric)
        "net_wealth_change": end_net_worth - start_net_worth,  # includes the ETH price move
        "controllable_gain": controllable_gain,                # interest + liquidation loss only
        "risk_adjusted_return": end_net_worth / (1 + liq_count),  # proposal: N_user / (1 + liquidation_count)
        "proactive_response_rate": (proactive_steps / elevated_risk_steps) if elevated_risk_steps > 0 else None,
        "invalid_action_rate": invalid_steps / len(episode_rewards),
        "danger_days_share": danger_days / len(episode_rewards),
        "epsilon": agent.epsilon,
        "target_net_active": agent.target_net_active,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train the DQN agent.")
    p.add_argument("--no-r", action="store_true", help="turn OFF the Risk Signal R")
    p.add_argument("--no-cs", action="store_true", help="turn OFF the Credit Score CS")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--buffer-penalty", type=float, default=0.0,
                   help="HF danger-zone penalty per day at HF 1.05 (0 = off, try 1.0)")
    return p.parse_args()


def main():
    args = parse_args()
    use_r, use_cs = not args.no_r, not args.no_cs
    run_name = f"{variant_name(use_r, use_cs)}_seed{args.seed}"
    if args.buffer_penalty > 0:
        run_name += f"_buf{args.buffer_penalty:g}"

    run_ckpt_dir = os.path.join(CHECKPOINT_DIR, run_name)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(run_ckpt_dir, exist_ok=True)

    print(f"=== Training run: {run_name} (R={'on' if use_r else 'off'}, "
          f"CS={'on' if use_cs else 'off'}, buffer_penalty={args.buffer_penalty:g}, "
          f"episodes={args.episodes}, split=train) ===")

    env = LendingEnv(seed=args.seed, use_risk_signal=use_r, use_credit_score=use_cs, split="train",
                     buffer_penalty=args.buffer_penalty)
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
                  f"avg_wealth_chg={np.mean([r['net_wealth_change'] for r in recent]):8.2f} | "
                  f"ctrl_gain={np.mean([r['controllable_gain'] for r in recent]):6.2f} | "
                  f"proactive={np.mean(proactive_vals) if proactive_vals else float('nan'):.2f} | "
                  f"danger_days={np.mean([r['danger_days_share'] for r in recent]):.2f} | "
                  f"invalid={np.mean([r['invalid_action_rate'] for r in recent]):.2f}")

        if ep % CHECKPOINT_EVERY == 0:
            torch.save(agent.online_net.state_dict(), os.path.join(run_ckpt_dir, f"ep{ep}.pt"))

    final_ckpt = os.path.join(run_ckpt_dir, "final.pt")
    torch.save(agent.online_net.state_dict(), final_ckpt)

    df = pd.DataFrame(log_rows)
    log_path = os.path.join(LOG_DIR, f"{run_name}.csv")
    df.to_csv(log_path, index=False)

    last = df.tail(100)  # last 100 episodes = the trained agent (epsilon is at its minimum)
    print(f"\nSaved log: {log_path}")
    print(f"Final checkpoint: {final_ckpt}")
    print(f"\nLast 100 episodes:")
    print(f"  liquidation rate:      {last['liquidated'].mean():.3f}")
    print(f"  avg reward:            {last['total_reward'].mean():.3f}")
    print(f"  avg net wealth change: {last['net_wealth_change'].mean():.2f}")
    print(f"  avg controllable gain: {last['controllable_gain'].mean():.2f}")
    print(f"  avg risk-adj. return:  {last['risk_adjusted_return'].mean():.2f}")
    print(f"  avg proactive rate:    {last['proactive_response_rate'].dropna().mean():.3f}")
    print(f"  avg invalid rate:      {last['invalid_action_rate'].mean():.3f}")
    print(f"  danger days (HF<{HF_BUFFER}):  {last['danger_days_share'].mean():.3f}")


if __name__ == "__main__":
    main()