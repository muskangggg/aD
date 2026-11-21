# auto_defi thesis: experiment journey, part 2 (6–7 Nov 2026)

Making the Risk Signal R and the Credit Score CS actually matter: what we did, why we did it, and what the results told us.

Part 1 of the journey (up to 30 Oct) is in the Project file claude/experiment_journey.md.


## 0. How to use this document

- **Friends / supervisor:** read sections 1–3 for the story, section 11 for the final results, section 12 for what we can claim.
- **A new chat (to keep improving R):** attach this file (or point to claude/experiment_journey_part2.md in the Project) and say "continue from section 15". Section 14 lists every file, run name and command.
- **Writing the thesis:** sections 4–11 follow the order of a Method → Results → Discussion chapter.


## 1. Short glossary

| Word | Meaning (simple) |
|---|---|
| HF (Health Factor) | How safe the loan is. HF 2.0 = very safe. HF below 1.0 = liquidation (the user loses money). |
| R (Risk Signal) | A number that warns about danger. In this chat it changed from "HF rescaled" to "chance of liquidation in the next 5 days". |
| CS (Credit Score) | A 0–1 score of the user's history, taken from 445 real Aave wallets. In this chat it changed from a reward weight to the user's borrow rate. |
| Agent groups | full = sees R and CS; r_only = sees only R; cs_only = sees only CS; none = sees neither (control). |
| Seed | One training run with its own randomness. 5 seeds = the same experiment repeated 5 times. |
| liq_rate | Share of 90-day episodes with at least one liquidation (lower is better). |
| ctrl_gain | Money the strategy controls: interest earned − interest paid + liquidation loss ($ per episode). |
| risk_adj | End net worth / (1 + number of liquidations). |
| pro_gap | How much MORE the agent protects the loan on risky days than on normal days (> 0 = it really reacts to risk). |
| CS→debt corr | Rank correlation between a user's CS and the agent's average debt. > 0 = high-CS users get bigger loans. |
| VAL / TEST / STRESS | VAL = tuning data; TEST = final data, used once; STRESS = hard synthetic markets (storms, crashes), only for testing. |


## 2. Where we started (6 Nov, morning)

After part 1 the agent was safe (0 liquidations in 2700 VAL episodes) thanks to the learning-rate fix (LR 1e-3 → 1e-4). The DQN setup was frozen: LR 1e-4, Double DQN, no Huber loss, gradient clip 10, hard target copy.

But the ablation (3 seeds) gave a result that hurts the thesis title:

- The agent that sees **neither R nor CS ("none") was the best and most stable**.
- **Adding CS made results worse** every time.
- **R did not make the agent more proactive.**

Goal of this chat: make the agent "more worthy of a thesis" by making R and CS matter, not by random upgrades.


## 3. Diagnosis: why R and CS did not help

We read the code to see what --no-r and --no-cs actually removed.

| Flag | What it removed (old env v3.3) |
|---|---|
| --no-r | health_deficit, volatility, utilization from the observation, AND the replay boost (PER). HF itself stayed visible. |
| --no-cs | CS from the observation, AND the m(CS) weight from the REWARD. |

- **R repeated HF.** R was about 50% health_deficit, and health_deficit is just HF rescaled. Example: being told the temperature in °C and then in °F — the second number adds nothing.
- **CS changed the goal, not the loan.** Every user had the same loan and the same rates; CS only made losses count more (×(1.5−CS)) or gains count more (×(0.5+CS)).

> **Key principle found in this chat: a feature can only help the agent if it changes WHAT THE BEST ACTION IS.**


## 4. Step 1: does CS make the old agents treat users differently?

**Why:** if CS works as designed, an agent that sees CS should keep low-CS users safer (higher HF, less debt) and let high-CS users borrow more. Agents that cannot see CS are the control group.

**How:** new script cs_analysis.py (no training). It splits the 300 VAL users into low / mid / high CS and measures HF, debt and money per group, plus a correlation per run.

**Result (old env, 3 seeds):**

- No consistent pattern: the full agent's CS→debt correlation was +0.14 ± 0.41 (signs flipped between seeds).
- CS agents were worse even on their OWN CS-weighted goal (full 46.3, none 71.3). So "CS is careful by design" was wrong.

**Why:** in the old env, taking more risk barely paid (extra borrowing earns about $0.50 per 90 days; one liquidation costs about $10–20 plus a −50 penalty). So "stay safe" was best for every user and CS had nothing to change. The m(CS) weight only added reward noise. Example: telling a waiter "VIP customers get bigger portions" when the kitchen has only one portion size.

**Decision:** Option B (approved by the supervisor): give CS a real job in the loan, and give R information HF does not have. Time left: 13 days to pre-defense.


## 5. Env v4: what we changed and why


### 5.1 CS sets the borrow rate

borrow APR = 2% + 6% × (1 − CS). Idle wallet USDC still earns 5%.

| CS | Borrow rate | Borrowing to hold USDC… |
|---|---|---|
| 0.25 | 6.5% | loses money → best: repay |
| 0.50 | 5.0% | break-even |
| 0.75 | 3.5% | earns money |
| 0.95 | 2.3% | earns money → best: keep / take the loan |

Example: a bank gives a cheap loan to a customer with a good history and an expensive loan to a risky one. Now the best action really depends on CS. (A first idea, 3% + 4% × (1 − CS), was too small an effect, so it was widened before any training.)

The m(CS) reward weight is switched OFF. Reward = 10 × (interest + liquidation loss) − 50 per liquidation.


### 5.2 R = chance of liquidation in the next 5 days

R = P(HF touches 1.0 within 5 days) = 2·Φ(−d / (σ·√5)), where d = log distance from today's ETH price to the liquidation price, and σ = today's daily volatility (RiskMetrics EWMA, λ = 0.94). Same idea as Merton's "distance to default". Works for every loan mix (ETH or USDC collateral / debt).

| HF | Calm (2%/day) | Normal (3.5%/day) | Stormy (7%/day) |
|---|---|---|---|
| 1.50 | 0.00% | 0.00% | 0.96% |
| 1.30 | 0.00% | 0.08% | 9.37% |
| 1.20 | 0.00% | 1.98% | 24.41% |
| 1.10 | 3.31% | 22.33% | 54.26% |

Example: two people 10 m from a cliff edge — one walking, one running. Same distance (HF), very different danger. HF shows the distance; R also shows the speed.


### 5.3 Calm and stormy markets + crashes (training data)

- Synthetic paths switch between CALM (0.6× train volatility) and STORMY (2×). Calm lasts ~30 days, storms ~10 days, so ~25% of synthetic days are stormy.
- One-day crashes of 10–25% happen with probability 1/90 per day (~1 per episode).
- Zero expected drift kept (no free up/down trend). Checked: mean 90-day change +0.3%.
- Training mix 50% real / 50% synthetic (was 70/30). VAL/TEST keep 70/30.
- New STRESS split: all synthetic, starts stormy, 3× more crashes. Only for testing.

Why: R uses volatility. If the market always moves at the same speed, knowing the speed is useless.


### 5.4 Clean ablation: hiding information only

--no-r hides R, volatility and utilization; --no-cs hides CS. The world stays the same for every group (the user still pays their CS-based rate; the market is still calm or stormy; the replay boost uses the true R for everyone). So any difference between groups comes only from what the agent can SEE.


### 5.5 Other changes

- New baseline **heuristic_cs**: the heuristic plus "never borrow if your rate is above 5%, pay down debt instead". A stronger rule that is GIVEN the CS formula.
- Run names end in _env4 (e.g. cs_only_seed42_env4) so old runs are never overwritten; training refuses to overwrite a finished run.
- PER high-risk threshold 0.7 → 0.2 (R is now a probability).
- "High risk" in the metrics = R > 5%.
- evaluate.py / cs_analysis.py got --seed (see section 10).

Unchanged: DQN setup, actions ($100 each), 1.05 safety rule, Aave-style liquidation, 90-day episodes, 1000 training episodes.


## 6. Self-tests (before any training)

| Check | Result |
|---|---|
| Action mask matches reality | 0 mismatches ✅ |
| Masked observation (no R, no CS) | last 4 values are 0; same user and rate ✅ |
| Low CS: repay-all vs keep loan | repay-all better (+4.92 vs +3.44 $) ✅ |
| High CS: repay-all vs keep loan | keep loan better (+7.63 vs +4.99 $) ✅ |
| R at HF 1.3, calm vs stormy | 0.08% vs 9.37% ✅ |
| Price drift / stormy share | +0.3% / 25% ✅ |


## 7. Pilot (1 seed, rule set BEFORE seeing results)

Rule: start the overnight run only if both agents are safe (liq ≤ 0.03) AND full uses CS (earns more than none, or shows a CS pattern).

|  | full_seed42 | none_seed42 | heuristic |
|---|---|---|---|
| liq_rate | 0.003 | 0.000 | 0.010 |
| money ($) | 4.29 | 3.31 | 4.83 |
| pro_gap | 0.815 | 0.528 | 0.494 |

Passed → overnight run started.


## 8. Overnight ablation

4 groups × 5 seeds (42, 1, 2, 3, 4) = 20 runs, 1000 episodes each (~7 hours). Then evaluated on VAL, STRESS and TEST, 300 episodes each, same price paths and users for every strategy.


## 9. VAL and STRESS results (mean ± std over 5 seeds)


#### VAL

| Group | liq_rate | money ($) | pro_gap | CS→debt |
|---|---|---|---|---|
| static | 0.030 | 4.54 | 0 | — |
| heuristic | 0.010 | 4.83 | 0.49 | — |
| heuristic_cs | 0.007 | 5.20 | 0.44 | +0.61 |
| full | 0.011 | 3.84 ± 1.04 | 0.63 | +0.02 |
| r_only | 0.025 | 1.79 ± 1.88 | 0.46 | −0.22 |
| cs_only | 0.005 | 4.77 ± 0.24 | 0.51 | +0.45 |
| none | 0.001 | 1.63 ± 5.82 | 0.32 | −0.17 |


#### STRESS (storms + 3× crashes)

| Group | liq_rate | money ($) | pro_gap |
|---|---|---|---|
| static | 0.260 | 1.15 | 0 |
| heuristic | 0.103 | 3.90 | 0.21 |
| heuristic_cs | 0.050 | 4.84 | 0.18 |
| full | 0.071 ± 0.052 | 2.09 ± 2.61 | 0.48 |
| r_only | 0.100 ± 0.083 | 0.10 ± 2.66 | 0.42 |
| cs_only | 0.050 ± 0.093 | 3.63 ± 1.95 | 0.40 |
| none | 0.046 ± 0.031 | 0.36 ± 6.42 | 0.25 |


## 10. TEST: a problem found and fixed

The first TEST run gave numbers almost identical to VAL (static $4.54 in both). Check: because both used the same evaluation seed (2026), VAL and TEST shared **all 300 users** and **90 of 300 synthetic price paths**; only the real price windows were new. Example: an exam with new reading texts but the same students and the same maths questions as the practice test.

Fix: a --seed option. TEST was re-run once with seed 3031 (new users, new paths). The agents were NOT changed, so this is not tuning. The first TEST run gave the same picture and is mentioned in one line in the thesis.


## 11. FINAL results: independent TEST (seed 3031)

| Group | liq_rate | risk_adj | money ($) | pro_gap | CS→debt |
|---|---|---|---|---|---|
| static | 0.043 | 944.3 | 4.43 | 0 | — |
| heuristic | 0.003 | 957.7 | 5.01 | 0.38 | — |
| heuristic_cs | 0.003 | 958.0 | 5.30 | 0.38 | +0.65 |
| full | 0.011 | 952.1 | 3.85 ± 1.13 | 0.48 | +0.11 |
| r_only | 0.014 | 949.0 | 2.25 ± 1.23 | 0.50 | −0.17 |
| cs_only | 0.007 | 955.0 | 4.73 ± 0.26 | 0.51 | +0.47 |
| none | 0.007 | 952.7 | 1.64 ± 5.55 | 0.19 | −0.11 |

cs_only average debt by CS group: low $281 → mid $308 → high $423 (control groups flat). 4 of 5 cs_only seeds show the pattern on VAL, TEST and TEST-3031.


## 12. What we can claim

| Claim | Status | Evidence |
|---|---|---|
| The agent keeps loans safe | ✅ | All learning groups liq ≤ ~1% on TEST |
| The agent is credit-aware (CS) | ✅ new | cs_only +0.47 corr, 4/5 seeds, control groups ≈ 0; most stable money (4.73 ± 0.26) |
| Design lesson | ✅ | CS as a reward weight did nothing; CS as a loan term is learned |
| R makes the agent proactive | 🟡 partly | none → r_only raises pro_gap on every split (TEST 0.19 → 0.50); with CS (cs_only → full) only on VAL/STRESS, not on TEST-3031 |
| R makes the agent safer / richer | ❌ | R agents had MORE liquidations in STRESS and earned less |
| Beats the hand-made rules on money | ❌ | heuristic 5.01, heuristic_cs 5.30 > cs_only 4.73 (but heuristic_cs is GIVEN the rule; the agent learned it) |
| Method: LR diagnosis (part 1) | ✅ | policy churn found and fixed |


## 13. Wrong guesses we corrected (be honest about these)

| What we first thought | What the evidence showed |
|---|---|
| CS agents earn less because they are careful on purpose | They were worse even on their own CS goal; CS was just noise |
| "All groups are safe, so the numbers can't differ" | Users can be safe at different levels; they were treated the same because extra risk barely paid |
| R adds proactiveness in every pairing | Only clearly without CS; on the independent TEST, full ≈ cs_only |
| The first TEST run was independent of VAL | It shared all users and 30% of price paths (same seed) |


## 14. Files, run names and commands

| Item | Where |
|---|---|
| New code (env v4) | Project: claude/env4/ (lending_env.py, compute_risk_signal.py, price_simulator.py, training_loop.py, evaluate.py, cs_analysis.py). Laptop: env/, features/, simulation/, main folder. |
| DQN change | dqn_agent.py: PRIORITY_HIGH_RISK_THRESHOLD = 0.2 (rest frozen) |
| Old code | git tag pre-env4 (old checkpoints only work with this code) |
| Results summaries | Project: claude/results_2026-11-06_cs_analysis.md, claude/results_2026-11-07_env4_ablation.md |
| Raw outputs (laptop) | results/eval_val_env4.txt, cs_val_env4.txt, eval_stress_env4.txt, eval_test_env4.txt, cs_test_env4.txt, eval_test_seed3031.txt, cs_test_seed3031.txt |
| Trained agents | checkpoints/<group>_seed<N>_env4/final.pt (20 runs) |
| Agent to deploy | cs_only_seed42_env4 (rule: best VAL group by risk_adj → best seed by VAL risk_adj). TEST-3031: liq 0.000, money 5.14, CS corr +0.74 |

```bash
# list of the 20 runs
RUNS=$(for v in full r_only cs_only none; do for s in 42 1 2 3 4; do echo -n "${v}_seed${s}_env4 "; done; done)
# train one run
python training_loop.py --seed 42 --no-r          # cs_only_seed42_env4
# evaluate
python evaluate.py --runs $RUNS --episodes 300                         # VAL
python evaluate.py --runs $RUNS --episodes 300 --split stress
python evaluate.py --runs $RUNS --episodes 300 --split test --seed 3031 # final, done once
python cs_analysis.py --runs $RUNS --episodes 300 [--split test --seed 3031]
# self-tests
python env/lending_env.py; python features/compute_risk_signal.py; python simulation/price_simulator.py
```


## 15. Open problems and ideas to improve R (start here in a new chat)

Current status of R: it makes agents react more (higher pro_gap), but not safer and not richer. Ideas, cheapest first:

1. **Diagnose first (no training):** in STRESS, check WHEN R agents get liquidated: on crash-jump days (no warning possible) or after slow storms (warning ignored). Also compare average HF of R vs no-R agents in stormy periods (are they sitting closer to the edge?).
2. **Bigger actions:** with $100 per day, an agent that sees high R cannot get safe fast enough. Try repay/deposit sizes like 10% / 25% of the debt.
3. **R as a safety shield:** if R > 20%, force a repay (a rule on top of the agent). Common in safe RL; good for Sepolia. Safety then partly comes from the rule — report it honestly.
4. **Jump-aware R:** add crash risk to the probability (e.g. a jump term), or a shorter horizon (1–2 days) in storms.
5. **Fewer inputs:** give only R (drop volatility and utilization) to make learning easier, and test R vs volatility-only.
6. **More seeds or longer training for full:** full learns CS in only 2/5 seeds; check if this is learning difficulty.
7. Other open work: small gas cost per action (the agent acts almost daily); policy.py for the friend's Sepolia service; none_seed1 failure (over-borrowing on every split).

> **Rule for any new experiment: set the decision rule before the run, change one thing at a time, use VAL for decisions, and do not touch TEST-3031 again (use a new TEST seed if a final re-check is ever needed).**


## 16. New deviations from the proposal (approved by the supervisor)

- CS acts through the loan (borrow rate) instead of the reward weight m(CS).
- R redefined as a 5-day liquidation probability (EWMA volatility).
- Regime-switching synthetic prices with crash jumps; 50/50 training mix; STRESS split.
- Info-only ablation (PER boost uses true R for every group).
- New baseline heuristic_cs; 5 seeds; independent TEST seed.

**Conclusion:** In this chat CS became a real, learned credit-aware behaviour (the thesis's strongest new result), R became a forward-looking warning that makes agents react more but not yet safer, and the next step for R is to find out why its warnings don't turn into fewer liquidations.
