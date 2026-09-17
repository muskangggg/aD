"""
training_loop.py

Step 8: the real training loop, on top of Step 7's DQNAgent.

Per-episode logging tracks the same 4 metrics Step 10's evaluation needs,
so you get a live read on training progress instead of waiting for the
ablation study to find out if it's working:

  - net_wealth_change      : net_worth_usd at episode end - at episode start
  - liquidated              : did the episode end in liquidation
  - risk_adjusted_return    : total_reward / (std of per-step rewards + eps)
                               (higher = more return per unit of reward volatility)
  - proactive_response_rate : of the steps where risk_signal was elevated
                               (> SAFE_RISK_THRESHOLD), what fraction did the
                               agent respond with a de-risking action
                               (deposit or repay, on either asset) rather
                               than hold/borrow/withdraw

Also saves a training_log.csv (one row per episode) and periodic model
checkpoints -- both needed for Step 9/10.

Usage:
    python training_loop.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))

from lending_env import LendingEnv, ACTION_TYPES, HOLD_ACTION, SAFE_RISK_THRESHOLD  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402

# ---- Training config (not specified by the proposal -- tune as needed) ----
N_EPISODES = 500          # ~45,000 steps at 90 steps/episode -- enough to see
                           # epsilon decay (20k steps) AND target net activation
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CHECKPOINT_EVERY = 100     # episodes
LOG_PRINT_EVERY = 20       # episodes

DE_RISKING_ACTION_TYPES = {"deposit", "repay"}


def _is_de_risking_action(action: int) -> bool:
    """True if this action is a deposit or repay (on any asset) -- the
    kinds of actions that reduce risk. Hold, borrow, and withdraw are not."""
    if action == HOLD_ACTION:
        return False
    action_type = ACTION_TYPES[action % len(ACTION_TYPES)]
    return action_type in DE_RISKING_ACTION_TYPES


def run_episode(env: LendingEnv, agent: DQNAgent) -> dict:
    obs, info = env.reset()
    start_net_worth = info.get("net_worth_usd")
    if start_net_worth is None:
        # reset() doesn't return info populated yet -- compute from balances directly
        start_net_worth = env.collateral_balance["ETH"] * env._asset_price("ETH") - env.debt_balance["USDC"]

    episode_rewards = []
    elevated_risk_steps = 0
    proactive_steps = 0
    liquidated = False
    end_net_worth = start_net_worth

    done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.store_transition(obs, action, reward, next_obs, terminated or truncated, info["risk_signal"])
        agent.learn()

        episode_rewards.append(reward)
        if info["risk_signal"] > SAFE_RISK_THRESHOLD:
            elevated_risk_steps += 1
            if _is_de_risking_action(action):
                proactive_steps += 1

        liquidated = info["liquidated"]
        end_net_worth = info["net_worth_usd"]

        obs = next_obs
        done = terminated or truncated

    total_reward = float(np.sum(episode_rewards))
    reward_std = float(np.std(episode_rewards)) + 1e-6
    risk_adjusted_return = total_reward / reward_std
    proactive_response_rate = (proactive_steps / elevated_risk_steps) if elevated_risk_steps > 0 else None

    return {
        "total_reward": total_reward,
        "episode_length": len(episode_rewards),
        "liquidated": liquidated,
        "net_wealth_change": end_net_worth - start_net_worth,
        "risk_adjusted_return": risk_adjusted_return,
        "proactive_response_rate": proactive_response_rate,
        "epsilon": agent.epsilon,
        "target_net_active": agent.target_net_active,
    }


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env = LendingEnv(seed=42)
    agent = DQNAgent(obs_dim=env.observation_space.shape[0], n_actions=env.action_space.n, seed=42)

    log_rows = []
    for ep in range(1, N_EPISODES + 1):
        metrics = run_episode(env, agent)
        metrics["episode"] = ep
        log_rows.append(metrics)

        if ep % LOG_PRINT_EVERY == 0:
            recent = log_rows[-LOG_PRINT_EVERY:]
            liq_rate = np.mean([r["liquidated"] for r in recent])
            avg_reward = np.mean([r["total_reward"] for r in recent])
            avg_wealth_change = np.mean([r["net_wealth_change"] for r in recent])
            proactive_vals = [r["proactive_response_rate"] for r in recent if r["proactive_response_rate"] is not None]
            avg_proactive = np.mean(proactive_vals) if proactive_vals else float("nan")

            print(f"ep {ep:4d} | eps={agent.epsilon:.3f} | target_active={agent.target_net_active} | "
                  f"avg_reward={avg_reward:8.3f} | liq_rate={liq_rate:.2f} | "
                  f"avg_wealth_chg={avg_wealth_change:8.2f} | proactive_rate={avg_proactive:.2f}")

        if ep % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"dqn_ep{ep}.pt")
            torch.save(agent.online_net.state_dict(), ckpt_path)

    # final checkpoint + full log
    torch.save(agent.online_net.state_dict(), os.path.join(CHECKPOINT_DIR, "dqn_final.pt"))
    df = pd.DataFrame(log_rows)
    log_path = os.path.join(LOG_DIR, "training_log.csv")
    df.to_csv(log_path, index=False)

    print(f"\nSaved {len(df)} episodes to {log_path}")
    print(f"Final checkpoint: {os.path.join(CHECKPOINT_DIR, 'dqn_final.pt')}")
    print(f"\nOverall liquidation rate: {df['liquidated'].mean():.3f}")
    print(f"Overall avg reward: {df['total_reward'].mean():.3f}")
    print(f"Overall avg net wealth change: {df['net_wealth_change'].mean():.2f}")


if __name__ == "__main__":
    main()