# auto_defi thesis: experiment journey, part 3 (18–21 Nov 2026)

Action counts, an action fee that failed, finding out WHY the Risk Signal R did not make agents safer, and a safety shield that fixed it.

Earlier parts: Part 1 (up to 30 Oct): claude/experiment_journey.md. Part 2 (6–7 Nov, env v4, CS works): claude/experiment_journey_part2.md.


## 0. How to use this document

- **Friends / supervisor:** read sections 1–2 for where we started, sections 9–9b for the final shield result, section 10 for what we can claim.
- **A new chat:** attach this file and say "continue from section 13". Section 12 lists every file, run name and command.
- **Writing the thesis:** sections 3–9b follow a Method → Results → Discussion order. The fee (section 4) and the plain shield (section 7) are "tried and rejected" ideas: report them briefly, they make the positive claims more believable.


## 1. Short glossary (new words only; see part 2 for HF, R, CS, groups, seeds)

| Word | Meaning (simple) |
|---|---|
| Action fee | A small $ cost for every action that does something (like gas on Ethereum). |
| Safety move (de-risk) | repay or deposit: makes the loan safer. |
| Risky move (re-risk) | borrow or withdraw: makes the loan riskier. |
| Undo / see-saw | A safety move followed soon by a risky move, so the loan ends up as risky as before. |
| Shield | A rule on top of the agent: while it is ON, borrow and withdraw are blocked. |
| Shield with memory | The shield turns ON when R > 5%, and only turns OFF when R < 1% AND at least 5 days have passed since R was last above 5%. |
| STRESS | Hard synthetic markets: start stormy, 3x more crashes. Never used for training. |
| Decision rule | The pass/fail rule we wrote down BEFORE running an experiment. |


## 2. Where we started (end of part 2)

- **CS works:** agents that see CS give low-CS users less debt and high-CS users more (cs_only, 4 of 5 seeds). Main positive result.
- **R did not make agents safer:** R agents reacted more to risk (higher pro_gap), but in STRESS they were liquidated MORE (r_only 0.100 vs none 0.046).
- **No learning group beats the hand-made rules on money** (TEST: heuristic_cs $5.30, heuristic $5.01, cs_only $4.73).
- Pre-defense: **5 Dec 2026**.


## 3. Step 1: which actions do the agents use? (18 Nov)

**Why:** the agent can borrow both ETH and USDC. Borrowing ETH is a bad move in our world: the borrowed ETH sits in the wallet earning 0% while the user pays interest (only wallet USDC earns 5%). A good agent should learn to skip it.

**How:** evaluate.py v3.2 got TABLE 3 = % of steps each of the 9 actions was chosen.

**Result (VAL, seed 2026):**

| agent | borrow_ETH | borrow_USDC / repay_USDC | hold |
|---|---|---|---|
| full_seed42 | 0.0% | 7.6% / 6.5% | 72% |
| none_seed42 | 5.5% | 34% / 33% | 12% |

- The agent with R and CS learned to skip the useless action and acts calmly.
- The blind agent flip-flops: borrow, repay, borrow, repay. Actions are free, so nothing stops it.

Side note (a correction we made): ETH debt does NOT push HF down when ETH rises in our setup, because the collateral is also ETH (the collateral grows more than the ETH debt). ETH debt is bad only because it costs interest and earns nothing.


## 4. Step 2: an action fee (18–19 Nov) -- tried and REJECTED

**Idea:** charge $0.05 for every action that does something (like gas), so flip-flopping costs money.

**Why $0.05 and not $0.50:** a $100 borrow for a high-CS user earns only ~$0.30 over a whole episode, and a $100 repay for a low-CS user saves only ~$0.20. A big fee would kill the useful actions too.

**Code:** env v4.1, `LendingEnv(action_fee=...)`, `training_loop.py --fee 0.05`. Default 0 = exactly v4 (self-test numbers identical). Everyone, including baselines, is evaluated in the same fee world.

**Decision rule (set before):** keep the fee if cs_only passes all 3 checks (liq_rate <= 0.03, holds more, ctrl_gain closer to heuristic_cs) in at least 2 of 3 seeds.

**Result (VAL, fee world, heuristic_cs = $5.00):**

| cs_only seed | liq_rate | hold % | ctrl_gain (old -> fee) | |
|---|---|---|---|---|
| 42 | 0.020 | 47 -> 84 | 2.77 -> 3.20 | pass |
| 1 | 0.063 | 25 -> 38 | 1.27 -> -4.68 | fail |
| 2 | 0.003 | 24 -> 1.5 | 1.13 -> -4.42 | fail |

- Only 1 of 3 seeds passed. cs_only money on average: +$1.72 (old agents) -> -$1.97 (fee agents). **Rejected.**
- none was mixed; none_seed42 learned a new bad habit (borrow ETH 22% + deposit that ETH 22%, debt $1958).
- Likely reasons: fee agents still acted on 38–57% of training days, and the fee also charges USEFUL repays, so some agents kept big debts instead.
- Example: a toll road meant to stop joyriders that also charges the ambulance.
- One side note, not a reason to keep it: with the fee, the CS pattern showed in 3 of 3 seeds (vs 2 of 3).


## 5. Step 3: WHY are R agents not safer? (19 Nov, no training)

**How:** new script diagnose_r.py. For the FIRST liquidation of each episode it looks at the 5 days before and sorts it into:

| cause | meaning | smoke-alarm example |
|---|---|---|
| crash_no_warning | R was low, then a one-day ETH drop of 10%+ | an explosion: no alarm possible |
| quiet_slide | R was low, no big drop | the alarm stayed silent in a slow fire |
| warned_ignored | R warned, the agent did no safety move | the alarm rang, you stayed on the sofa |
| warned_too_slow | R warned, the agent did safety moves, still liquidated | you ran for water but only had a cup |

**Result (STRESS, seed 2026, 300 episodes, 20 env4 agents), share of each group's liquidations:**

| group | crash_no_warning | warned_ignored | warned_too_slow |
|---|---|---|---|
| full | 42% | 10% | 47% |
| r_only | 32% | 3% | 63% |
| cs_only | 56% | 4% | 40% |
| none | 29% | 50% | 21% |

quiet_slide was ~0% for everyone: R almost never misses a slow danger.

**What we learned:**
1. **R works as an alarm:** agents that see R almost never ignore it (r_only 3%, full 10%) vs none 50%.
2. **$100 per day is NOT too slow:** on the first warning day, the median repays needed to get R under 5% was 0.3–0.5 (less than ONE $100 repay), with 4–5 warning days and $300–700 in the wallet. So "bigger actions" (an idea from part 2) would not help.
3. **R agents do NOT sit closer to the edge:** average HF in storms full 2.18, r_only 2.16 vs none 1.82.
4. **Crashes are 30–55% of learning-agent liquidations:** no volatility-based R can see those coming.
5. **Bad seeds drive the group means:** all liquidations per seed, r_only 1/19/20/70/75, none 1/22/26/25/29. By median, r_only is not worse than none.


## 6. Step 4: do the agents UNDO their safety moves? (19 Nov, diagnose_r.py v1.1)

**Why:** "warned_too_slow" was the main cause for R agents, but less than one repay would have been enough. So maybe they repaid and then borrowed again.

**How:** for every warned liquidation, count inside the warning window (first R > 5% day -> liquidation) the safety moves, the risky moves, and the change in debt.

| group | undid (both kinds of move) | safety vs risky moves | debt change |
|---|---|---|---|
| r_only | **83%** | 1.98 vs 2.02 | +$0.28 (not down) |
| full | **52%** | 1.31 vs 1.15 | +$0.12 (not down) |
| cs_only | 76% | 1.91 vs 1.67 | -$99 (really paid down) |
| none | 20% | 0.54 vs 0.38 | $0 |
| heuristic | 17% | 0.61 vs 0.43 | ~$0 |

**Confirmed: R agents undo their safety moves.** r_only repays about 2 times and borrows about 2 times, so the debt does not change (HF only 1.16 -> 1.19). Example: one step back from the cliff, then one step forward again.

Likely reasons (not proven): borrowing gives a small reward every day (wallet USDC earns 5%), liquidation is rare in training, and repay and borrow have almost equal value for the agent, so a small change in R flips its choice.


## 7. Step 5: a plain shield (19 Nov) -- tried and REJECTED

**Idea:** a rule on top of the agent: while R > 5%, block borrow and withdraw. The DQN picks its best remaining action; hand-made rules fall back to hold. No retraining. It uses the TRUE R for every agent (also agents that cannot see R).

**Decision rule (set before):** keep if on STRESS the median liq_rate over 5 seeds drops >= 30% for BOTH full and r_only, AND on VAL no group loses more than $0.50.

**Result (STRESS, seed 2026):** median liq_rate full 0.077 -> 0.067 (-13%), r_only 0.063 -> 0.063 (0%). VAL unchanged. **Rejected.**

**Why (hypothesis, later confirmed by step 6):** a see-saw around the line. The agent repays, R drops from 6% to 4%, the shield switches off, the agent borrows again. Example: a parent who takes the car keys only while it is raining and gives them back the second the rain stops.


## 8. Step 6: a shield WITH MEMORY (20 Nov) -- PASSED

**Change:** ON when R > 5%; OFF only when R < 1% AND at least 5 days have passed since R was last above 5% (`--shield 0.05 --shield-off 0.01 --shield-days 5`). Example: the parent keeps the keys until the sky is really clear AND the warning is 5 days old.

**Self-test:** on a made-up strategy that does exactly the see-saw, liquidations went 0.667 -> 0.367 (plain shield: no help).

**Result (same rule as step 5, STRESS seed 2026):**

| group | median liq (none -> plain -> memory) | median drop | mean liq | STRESS money | VAL money |
|---|---|---|---|---|---|
| full | 0.077 -> 0.067 -> 0.053 | **-31%** | 0.071 -> 0.055 | 2.09 -> 2.40 | 3.84 -> 3.81 |
| r_only | 0.063 -> 0.063 -> 0.040 | **-37%** | 0.100 -> 0.053 | 0.10 -> 1.03 | 1.79 -> 1.88 |
| cs_only | 0.007 -> 0.007 -> 0.003 | -57% | 0.050 -> 0.034 | 3.63 -> 3.89 | 4.77 -> 4.77 |
| none | 0.043 -> 0.043 -> 0.050 | +16% | 0.046 -> 0.043 | 0.36 -> 0.55 | 1.63 -> 1.64 |

**Passed** (full only just over 30%). No money cost on VAL, more money on STRESS.

**The diagnosis and the fix agree:** the shield helped most where step 6 found the most undoing (mean liq drop: r_only 47%, cs_only 31%, full 22%, none 8%; undo rates 83%, 76%, 52%, 20%).


## 9. Step 7: confirmation on a NEW seed (21 Nov)

**Why:** we tried 2 shield versions on the same STRESS seed (2026), so the pass could be partly luck.

**Decision rule (set before):** confirmed if on STRESS seed 4049 the median liq_rate drops >= 20% for both full and r_only.

| group | median liq no shield -> memory shield | median drop | mean liq | money |
|---|---|---|---|---|
| full | 0.073 -> 0.043 | **-41%** | 0.077 -> 0.048 | 1.85 -> 2.52 |
| r_only | 0.063 -> 0.033 | **-48%** | 0.093 -> 0.044 | -0.02 -> 1.23 |
| cs_only | 0.007 -> 0.003 | -57% | 0.043 -> 0.029 | 3.80 -> 4.12 |
| none | 0.043 -> 0.040 | -7% | 0.042 -> 0.039 | 0.47 -> 0.60 |

**Confirmed.** Same pattern as seed 2026: big help for agents that undo, almost none for none. Deployment agent cs_only_seed42: 0 liquidations with and without the shield. Baselines: heuristic 0.073 -> 0.067, heuristic_cs unchanged.


## 9b. Step 8: FINAL TEST run with the shield (21 Nov, run once)

**Rule (set before):** safety holds if the median liq_rate drops >= 20% for full AND r_only; no money cost if no group's mean ctrl_gain drops > $0.50; if the no-shield median of BOTH full and r_only is < 0.02, TEST is too calm to measure safety and only the money check is reported. Report whatever comes out.

**Result (TEST seed 3031, 300 episodes):** no-shield medians full 0.010, r_only 0.007 -> **too calm; safety not measurable on TEST.**

| group | liquidated episodes (5 seeds) | mean liq_rate | mean ctrl_gain |
|---|---|---|---|
| full | 17 -> 12 | 0.011 -> 0.008 | 3.85 -> 3.92 |
| r_only | 21 -> 16 | 0.014 -> 0.011 | 2.25 -> 2.34 |
| cs_only | 10 -> 9 | 0.007 -> 0.006 | 4.73 -> 4.72 |
| none | 10 -> 11 | 0.007 -> 0.007 | 1.64 -> 1.61 |

- **Money check PASSED** (largest drop $0.03). Shield ON on only 0.1–8% of steps.
- Deployment agent cs_only_seed42: 0 liquidations both, $5.14 -> $5.13. Baselines unchanged (heuristic_cs $5.30).
- Direction matches STRESS, but too small to claim. Example: an airbag tested on a quiet road: we cannot measure how much it helps, but we see it never fires by mistake.
- Details: claude/results_2026-11-21_test_shield.md.


## 10. What we can claim (updated)

| Claim | Status | Evidence |
|---|---|---|
| The agent is credit-aware (CS) | ✅ (part 2) | cs_only +0.47 corr, 4/5 seeds, TEST-3031 |
| The agent learns to skip a useless action | ✅ new | full_seed42 borrow_ETH 0.0% vs none 5.5% |
| R is a correct early warning | ✅ new | quiet_slide ~0%; R agents ignore warnings 3–10% vs none 50% |
| R as an agent INPUT makes agents safer | ❌ | R agents react, then undo (r_only 83%) |
| R as a SAFETY LAYER (shield with memory) makes agents safer | ✅ new | STRESS liq -31%/-37% (seed 2026), -41%/-48% (seed 4049) for full/r_only, no VAL money cost |
| The shield costs nothing in normal markets | ✅ new | TEST-3031: largest group money drop $0.03; safety not measurable there (too few liquidations) |
| An action fee stops churn | ❌ | 1 of 3 seeds, money worse |
| Bigger actions are needed | ❌ (not needed) | < 1 repay needed at first warning |
| Beats the hand-made rules on money | ❌ | heuristic_cs still higher |
| Method: diagnose, fix, confirm with pre-set rules | ✅ | sections 5–9b |

**How to say the shield result honestly:** "learned agent + R-based safety layer". The safety gain comes from a rule driven by R, not from the agent learning it. The plain shield did not work; the memory (wait until it is really safe) is what matters. Crash jumps stay unavoidable. The safety gain is shown on STRESS; TEST was too calm to measure it and only shows no cost.


## 11. Wrong guesses we corrected (be honest about these)

| What we first thought | What the evidence showed |
|---|---|
| ETH debt pushes HF down when ETH rises | Not with ETH collateral: HF still goes up |
| A $0.50 fee is a good start | Far too big: a useful borrow earns only ~$0.30 per episode |
| An action fee will stop churn | Worked in 1 of 3 seeds; it also punished useful repays |
| $100 per day is too slow once R warns | Less than one repay was needed; the agents undid it instead |
| R agents sit closer to the edge | No: their storm HF was higher than none's |
| "R agents are less safe" is a property of R | Mostly 2 bad r_only seeds (3, 4); by median r_only ≈ none |
| Blocking risky moves while R > 5% is enough | No: the see-saw just moves below the line; memory is needed |


## 12. Files, run names and commands

| Item | Where |
|---|---|
| Code (Project: claude/env4/) | lending_env.py v4.1 (action_fee, default 0), training_loop.py v4.1 (--fee), evaluate.py v3.5 (TABLE 3 action counts, --fee, --shield, --shield-off, --shield-days), cs_analysis.py v2.1 (--fee), diagnose_r.py v1.1 (new) |
| Result summaries (Project) | claude/results_2026-11-19_fee_and_r_diagnosis.md, claude/results_2026-11-19_r_shield.md, claude/results_2026-11-21_test_shield.md |
| Raw outputs (laptop, results/) | eval_val_fee0.05_3seeds.txt, cs_val_fee0.05_3seeds.txt, diagnose_r_stress.txt, diagnose_r_stress_v1.1.txt, eval_stress_shield0.05.txt, eval_val_shield0.05.txt, eval_stress_shieldmem.txt, eval_val_shieldmem.txt, eval_stress_seed4049.txt, eval_stress_seed4049_shieldmem.txt, eval_test_seed3031.txt, eval_test_seed3031_shieldmem.txt (+ matching .csv files) |
| Fee agents (not used) | checkpoints/{cs_only,none}_seed{42,1,2}_env4_fee0.05/ |
| git | tags pre-fee (env v4 + action table), shield-confirmed; commit "fee experiment: failed rule, not used" |

```bash
RUNS=$(for v in full r_only cs_only none; do for s in 42 1 2 3 4; do echo -n "${v}_seed${s}_env4 "; done; done)
# diagnosis (no training)
python diagnose_r.py | tee results/diagnose_r_stress_v1.1.txt
# shield with memory
python evaluate.py --runs $RUNS --episodes 300 --split stress --shield 0.05 --shield-off 0.01 --shield-days 5
python evaluate.py --runs $RUNS --episodes 300 --shield 0.05 --shield-off 0.01 --shield-days 5          # VAL
python evaluate.py --runs $RUNS --episodes 300 --split stress --seed 4049 [--shield ...]                 # confirmation
# final TEST (done 21 Nov; do not re-run for tuning)
python evaluate.py --runs $RUNS --episodes 300 --split test --seed 3031 [--shield 0.05 --shield-off 0.01 --shield-days 5]
# fee (rejected; for the record)
python training_loop.py --seed 42 --no-r --fee 0.05
python evaluate.py --runs ... --fee 0.05
```


## 13. Open problems and ideas (start here in a new chat)

1. **Train WITH the shield on** (cheapest real follow-up): if the shield is active during training too, the agent may learn not to undo, and the safety would then partly be learned, not only a rule. Needs new runs (~7 h for 20 runs) and a decision rule set first.
2. **Crash-aware R:** 30–55% of liquidations are one-day crashes; a jump term in R (or a bigger safety buffer in storms) is the only way to touch those.
3. **Seed instability:** full_seed1/2, r_only_seed4, cs_only_seed1, none_seed1 are weak on every split. More seeds or longer training would make the claims stronger.
4. **Money:** no learning group beats heuristic_cs on money yet.
5. **TEST: DONE (21 Nov).** Final TEST with the shield was run once (section 9b). TEST-3031 is now closed; do not use it again.
6. **Sepolia / friend's part:** policy.py for the live service should include the shield with memory (it only needs R, which the service already computes).

> **Rule for any new experiment (unchanged): set the decision rule before the run, change one thing at a time, use VAL/STRESS for decisions, report failures too.**


## 14. New deviations from the proposal (to tell the supervisor)

- An optional action fee was added to the environment (tested, not used; default 0).
- An R-based safety shield with memory was added as an evaluation-time layer on top of the agent. It extends the proposal's 1.05 safety rule (Sec 5.1) from "block actions that break HF 1.05" to "block risky actions while the forecast risk R is high, until it is clearly safe again".
- A new STRESS seed (4049) was used to confirm the shield.

**Conclusion:** In part 3 we found that R is a correct alarm but the agents undo their own safety moves, a simple fee and a plain shield did not fix this, and a shield with memory cut liquidations in hard markets by about 30–50% on two seeds at no money cost (also no cost on the final TEST), which is the second positive result of the thesis next to credit-aware CS.
