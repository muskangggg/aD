"""
test_policy.py  (v1.0 -- checks that policy.py decides EXACTLY like evaluate.py v3.5)

Put it in the same folder as evaluate.py and policy.py, then:
    python test_policy.py                                   # cs_only_seed42_env4, VAL, 50 episodes
    python test_policy.py --run full_seed42_env4 --split stress --episodes 100

Checks:
  A. policy.py constants = env/lending_env.py constants (actions, sizes, HF rules, scaling).
  B. Replay in the simulator, WITH and WITHOUT the shield: on every step the
     live policy only gets what the service will have (token amounts, ETH
     price, CS) and must give the same observation, the same R, the same
     action mask and the SAME ACTION as evaluate.py. Mismatches should be 0.
  C. Shield memory timeline on a made-up R series: ON at R > 5%, stays ON
     until R <= 1% AND 5 days have passed since the last alarm.
  D. Live-style use (reset(), warm_up(), decide()) runs and returns a valid action.

Extra: with --episodes 300 on VAL (seed 2026) the liq_rate / ctrl_gain printed
for "shield ON" should equal the cs_only_seed42_env4 row of
results/eval_val_shieldmem.txt (same paths, same users).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "env"))

import lending_env as le  # noqa: E402
import policy as pl  # noqa: E402
from evaluate import make_dqn_policy, risk_before_action, shield_mask as ref_shield_mask  # noqa: E402


def check_a() -> bool:
    pairs = {
        "ASSETS": (pl.ASSETS, le.ASSETS), "ACTION_TYPES": (pl.ACTION_TYPES, le.ACTION_TYPES),
        "HOLD_ACTION": (pl.HOLD_ACTION, le.HOLD_ACTION),
        "LIQUIDATION_THRESHOLD": (pl.LIQUIDATION_THRESHOLD, le.LIQUIDATION_THRESHOLD),
        "ACTION_SIZE_USD": (pl.ACTION_SIZE_USD, le.ACTION_SIZE_USD),
        "MIN_HF_AFTER_ACTION": (pl.MIN_HF_AFTER_ACTION, le.MIN_HF_AFTER_ACTION),
        "EPISODE_LENGTH": (pl.EPISODE_LENGTH, le.EPISODE_LENGTH),
        "DAYS_PER_YEAR": (pl.DAYS_PER_YEAR, le.DAYS_PER_YEAR), "USD_SCALE": (pl.USD_SCALE, le.USD_SCALE),
        "HF_CLIP": (pl.HF_CLIP, le.HF_CLIP), "VOL_NORM_SCALE": (pl.VOL_NORM_SCALE, le.VOL_NORM_SCALE),
        "SHIELD_ON_R vs SAFE_RISK_THRESHOLD": (pl.SHIELD_ON_R, le.SAFE_RISK_THRESHOLD),
    }
    bad = [k for k, (a, b) in pairs.items() if a != b]
    print(f"Check A - constants: {len(pairs) - len(bad)}/{len(pairs)} match"
          + (f"  MISMATCH: {bad}" if bad else "  (OK)"))
    return not bad


def position_of(env) -> dict:
    return {"wallet_ETH": env.wallet["ETH"], "wallet_USDC": env.wallet["USDC"],
            "collat_ETH": env.collateral["ETH"], "collat_USDC": env.collateral["USDC"],
            "debt_ETH": env.debt["ETH"], "debt_USDC": env.debt["USDC"]}


def check_b(run, split, seed, episodes, use_shield) -> bool:
    use_r, use_cs = pl.VARIANT_FLAGS[run.rsplit("_seed", 1)[0]]
    env = le.LendingEnv(seed=seed, split=split, use_risk_signal=use_r, use_credit_score=use_cs)
    ref_policy = make_dqn_policy(run, 12, 9)
    sigma0 = env.price_sim.sigma_daily
    shield = pl.SHIELD_ON_R if use_shield else None
    live = pl.LivePolicy(run, shield=shield, shield_off=pl.SHIELD_OFF_R, shield_days=pl.SHIELD_DAYS,
                         sigma_daily_init=sigma0)

    steps = act_mis = mask_mis = 0
    obs_err = r_err = 0.0
    liq_eps, ctrl_total, shielded = 0, 0.0, 0
    first_bad = None
    for ep in range(episodes):
        obs, info = env.reset()
        live.reset(start_price=env.start_price, sigma_daily=sigma0)
        shield_on, last_alarm, day = False, -10 ** 9, 0              # evaluate.py v3.5 shield
        done, ctrl = False, 0.0
        while not done:
            _, r_now = risk_before_action(env)
            if shield is not None:
                if r_now > shield:
                    shield_on, last_alarm = True, day
                elif shield_on and r_now <= pl.SHIELD_OFF_R and day - last_alarm >= pl.SHIELD_DAYS:
                    shield_on = False
            day += 1
            if shield_on:
                a_ref = int(ref_policy(obs, {**info, "action_mask": ref_shield_mask(info["action_mask"])}))
            else:
                a_ref = int(ref_policy(obs, info))

            d = live.decide(env._price("ETH"), position_of(env), env.credit_score)

            steps += 1
            shielded += int(d.shield_on)
            obs_err = max(obs_err, float(np.max(np.abs(np.asarray(d.obs) - obs))))
            r_err = max(r_err, abs(d.R - r_now))
            mask_bad = not np.array_equal(np.asarray(d.mask), info["action_mask"])
            mask_mis += int(mask_bad)
            if d.action != a_ref or d.shield_on != shield_on:
                act_mis += 1
                if first_bad is None:
                    first_bad = (ep, day, a_ref, d.action, shield_on, d.shield_on, r_now)

            obs, _, terminated, truncated, info = env.step(a_ref)
            ctrl += info["interest_usd"] + info["liquidation_loss_usd"] - info["fee_usd"]
            done = terminated or truncated
        liq_eps += int(info["liquidation_count"] > 0)
        ctrl_total += ctrl

    ok = act_mis == 0 and mask_mis == 0 and obs_err < 1e-5 and r_err < 1e-9
    label = "shield ON " if use_shield else "shield off"
    print(f"Check B - {run}, {split.upper()} seed {seed}, {episodes} episodes, {label}: "
          f"{steps} steps, action mismatches {act_mis}, mask mismatches {mask_mis}, "
          f"max obs diff {obs_err:.2e}, max R diff {r_err:.2e} -> {'OK' if ok else 'FAIL'}")
    print(f"          liq_rate {liq_eps / episodes:.3f}, ctrl_gain {ctrl_total / episodes:.2f}, "
          f"shield ON on {shielded / steps:.1%} of steps")
    if first_bad:
        ep, day, a_ref, a_live, s_ref, s_live, r = first_bad
        print(f"          first mismatch: episode {ep}, day {day}, R {r:.4f}: evaluate.py {a_ref} "
              f"(shield {s_ref}) vs policy.py {a_live} (shield {s_live})")
    return ok


def check_c() -> bool:
    pol = pl.LivePolicy("full_seed42_env4", q_function=lambda o: np.zeros(9), sigma_daily_init=0.035)
    r_series = [0.06, 0.03, 0.005, 0.005, 0.005, 0.005, 0.005, 0.07, 0.02, 0.0]
    expected = [True, True, True, True, True, False, False, True, True, True]
    got = [pol._update_shield(r) for r in r_series]
    ok = got == expected
    print(f"Check C - shield memory: R {r_series}\n          ON {got} -> {'OK' if ok else 'FAIL'} "
          f"(ON at R > 5%; OFF on day 6 = R <= 1% and 5 days after the alarm)")
    return ok


def check_d(run) -> bool:
    pol = pl.LivePolicy(run)
    pol.warm_up([3000, 2950, 3050, 3000, 2900, 2980])
    price = 3000.0
    pos = {"wallet_ETH": 0.0, "wallet_USDC": 400.0, "collat_ETH": 1000.0 / price,
           "collat_USDC": 0.0, "debt_ETH": 0.0, "debt_USDC": 400.0}
    d = pol.decide(price, pos, 0.6)
    ok = 0 <= d.action <= 8 and d.mask[d.action]
    print(f"Check D - live-style call: {d.action_name} {d.amount:.4f} {d.asset or ''}, HF {d.health_factor:.2f}, "
          f"R {d.R:.3%}, shield {'ON' if d.shield_on else 'off'} -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=pl.DEFAULT_RUN)
    p.add_argument("--split", default="val", choices=["val", "stress", "train"],
                   help="never test on 'test' (TEST seed 3031 is closed)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--episodes", type=int, default=50)
    args = p.parse_args()

    results = [check_a(),
               check_b(args.run, args.split, args.seed, args.episodes, use_shield=True),
               check_b(args.run, args.split, args.seed, args.episodes, use_shield=False),
               check_c(),
               check_d(args.run)]
    print(f"\n{'ALL CHECKS PASSED' if all(results) else 'SOME CHECKS FAILED'} ({sum(results)}/{len(results)})")


if __name__ == "__main__":
    main()