"""
test_liquidation_reward.py

Quick one-off check (not part of the pipeline) that deliberately drives a
position into liquidation, to confirm LIQUIDATION_PENALTY actually swamps
the reward as intended. Random actions rarely liquidate on their own, so
this forces it by repeatedly borrowing the debt-side asset (USDC).

Usage:
    python test_liquidation_reward.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
from lending_env import LendingEnv, ACTION_TYPES, ASSETS  # noqa: E402

# action index for "borrow USDC": asset_idx=1 (USDC), action_type_idx=2 (borrow)
BORROW_USDC_ACTION = ASSETS.index("USDC") * len(ACTION_TYPES) + ACTION_TYPES.index("borrow")


def main():
    env = LendingEnv(seed=7)
    obs, info = env.reset()
    print(f"Sampled credit score: {env.credit_score:.3f}\n")

    for step_i in range(90):
        obs, reward, terminated, truncated, info = env.step(BORROW_USDC_ACTION)
        print(f"step {step_i}: reward={reward:.4f} HF={info['health_factor']:.3f} "
              f"risk_signal={info['risk_signal']:.3f} liquidated={info['liquidated']}")
        if terminated:
            print(f"\nLiquidated at step {step_i}. Final reward this step: {reward:.4f}")
            break
        if truncated:
            print("\nEpisode ended without liquidation (increase episode length or borrow size).")
            break


if __name__ == "__main__":
    main()