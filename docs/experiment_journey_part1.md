# auto_defi thesis: the full experiment journey (up to 30 Oct 2026)

This file explains, in order, **everything we tried** to make the DQN agent learn, **why** we tried it,
**what happened**, **what problem we found next**, and **what we decided**.
Use it to understand your own work and to write the Methodology / Results / Discussion chapters.

Raw numbers: `results/results_2026-10-30.txt`, `results/eval_val_ablation.txt`.
Current status and commands: `HANDOFF_left_part_status.md`.

---

## 1. The goal in short

- **Agent:** a DQN agent that manages ONE user's loan on an Aave-style lending pool (ETH collateral, USDC debt).
- **Main thesis claim:** the agent is **proactive**: it protects the loan **before** it becomes unsafe (HF < 1.0),
  helped by a **Risk Signal R** in its input. A **Credit Score CS** makes it credit-aware.
- **Compared against:** static (do nothing), heuristic (repay if HF < 1.2, borrow if HF > 2.0), random, borrow_all.
- **Final goal:** run the agent on the Sepolia testnet (friend builds the live service).

**Health Factor (HF)** = how safe the loan is. HF 2.0 = very safe. HF < 1.0 = liquidation (you lose money).

---

## 2. Metrics (what the numbers mean)

| metric | meaning | good value |
|---|---|---|
| liq_rate | share of episodes with at least one liquidation | 0 |
| risk_adj | end net worth / (1 + number of liquidations) | high (heuristic 784.3) |
| ctrl_gain | money the strategy controls: interest earned − interest paid + liquidation loss | high (heuristic 7.23, static 6.48) |
| pro_gap | how much MORE the agent de-risks on risky days than on normal days | > 0.1 (heuristic 0.104) |
| high_risk | share of days spent in high risk (R > 0.5, about HF < 1.3) | low (heuristic 0.097) |
| hf_lt_1.1 | share of days very close to liquidation | low |
| over-guess (gap) | how far the agent's predicted value is from the real outcome | small |

---

## 3. All approaches at a glance

| # | Phase | Approach | Why we tried it | Result | Kept? |
|---|---|---|---|---|---|
| A1-A10 | Setup | Environment and data fixes (bugs, splits, reward, masking...) | the simulator had to be correct and fair first | correct, fair setup | ✅ |
| B1 | Measure | evaluate.py v2 (pro_gap, borrow_all, interest split) | could not see WHY agents failed | could see behaviour and money flows | ✅ |
| C1 | Learn | Raise ALPHA | make earning signal louder | skipped: changes nothing in relative terms | ❌ |
| C2 | Learn | Double DQN + Huber + grad clip (v3h) | stop over-borrowing | interest fixed, but safety lost (liq 0.357) | partly |
| C3 | Learn | Huber off (v3) | get fear of liquidation back | liq 0.253 | ✅ Double + clip kept |
| C4 | Learn | Day-90 fix + 3000 episodes (v4) | "maybe it is under-trained" | worse (liq 0.263, ctrl −0.33) | fix kept, long training rejected |
| C5 | Reward | Danger-zone penalty (buf1) | make sitting near HF 1.05 costly | worse (liq 0.550) | ❌ off |
| D1 | Diagnose | diagnose.py Checks 1-3 (value guesses) | find what blocks learning | values far too high (+62 to +144) | ✅ tool |
| D2 | Diagnose | Check 0 (every checkpoint) | is the error growing? | agent SWINGS safe ↔ unsafe | ✅ key finding |
| D3 | Diagnose | Evaluate good checkpoints (ep300/ep500) | were they really safe? | 0.073 / 0.057: lucky snapshots | info |
| E1 | Learn | **LR 1e-3 → 1e-4** | updates too big → choices flip | **0/300 liquidations, stable** | ✅ **main fix** |
| E2 | Check | LR 1e-4 on seeds 1 and 2 | lucky seed? | 0/900 at the end, small swings left | ✅ |
| E3 | Learn | Soft target updates | remove remaining jumps | no clear gain | ❌ off (flag kept) |
| F1 | Ablation | full / r_only / cs_only / none × 3 seeds | does R / CS help? | simplest agent (none) is best | open question |

---

## 4. Phase A: building a correct, fair setup (before 29 Oct)

These were bugs or design problems. Each one had to be fixed before any result could be trusted.

| # | Problem | What we changed | Why it matters |
|---|---|---|---|
| A1 | **Free-money bug**: actions could create money that did not exist | Real wallet balances; an action only works if the money is there | Old results (23 Oct) are NOT comparable for this reason |
| A2 | Ablation needed (with / without R and CS) | Masking: R parts / CS are hidden from the observation | Allows the ablation study |
| A3 | Inputs on very different scales (dollars vs ratios) | Observation normalised (USD/1000, HF clipped at 5 and /5, ...) | Neural networks learn badly with unscaled inputs |
| A4 | Testing on the same data as training | Split by time: train 700 days, val 100 days, test 200 days | Fair test on unseen market periods; TEST used once at the end |
| A5 | Old reward only punished risk, never rewarded earning | Proposal reward: interest + liquidation loss, weighted by CS; −BETA per liquidation; 1.05 safety rule | Agent must balance earning and safety |
| A6 | Risk signal only warned when it was too late | health_deficit made an early warning (0 at HF ≥ 1.5, 1 at HF 1.0) | Needed for "proactive" |
| A7 | Agent too short-sighted; penalties too weak/strong | GAMMA 0.5 → 0.95; BETA 5 → 50; ALPHA 1 → 10 | Liquidation risk builds over days, so the agent must look ahead |
| A8 | Agent wasted time on actions that do nothing | Action masking (Huang & Ontanon 2020) | Only possible actions are chosen and learned |
| A9 | Episode ended at liquidation, so the agent never learned what comes after | Liquidation no longer ends the episode (continues to day 90) | More realistic, more learning data |
| A10 | Simulated prices drifted up or down unrealistically | GBM with zero drift; 70% real price windows + 30% GBM | No built-in "free" trend |

---

## 5. Phase B: better measurement (29 Oct)

**B1. evaluate.py v2**
- **Problem:** we only saw "liquidated or not". We could not see if the agent truly reacted to risk, or where money went.
- **What we added:** pro_gap, a borrow_all baseline (only the 1.05 rule protects it), interest earned/paid split, risk measured BEFORE the action.
- **Found:** v2 agent pro_gap 0.30 (heuristic 0.10); the 1.05 rule alone liquidated 100% of episodes; the v2 agent paid 3x more interest (over-borrowing).
- **Decision:** keep v2 evaluation for every later test. Same 300 VAL price paths for every strategy.

---

## 6. Phase C: trying to make the agent learn (29 Oct)

**Starting point, v2 (plain DQN, LR 1e-3, 1000 episodes):** liq 0.003 (looked safe), but ctrl_gain only 1.36,
debt $1502, interest paid 9.76. Later diagnosis showed its safety came from an **accidental hedge** (borrowing ETH), not from understanding risk.

**C1. Raise ALPHA (50 or 100)** — skipped
- **Idea:** make the earning signal louder.
- **Why skipped:** ALPHA multiplies interest AND liquidation loss, so everything gets louder by the same amount. Nothing changes in relative terms.

**C2. Double DQN + Huber loss + gradient clipping (v3h)**
- **Problem:** over-borrowing, too much interest paid.
- **Why these:** plain DQN over-estimates noisy actions; Double DQN (van Hasselt 2016) splits "pick" and "judge"; Huber limits big errors; clipping limits big steps.
- **Result:** interest paid 9.76 → 4.59, debt 1502 → 439 ✅, but liq 0.003 → **0.357** ❌.
- **Issue found:** Huber made big liquidation losses "quieter", so the agent forgot to fear them.
- **Decision:** keep Double DQN and clipping, test without Huber.

**C3. Huber off (v3)**
- **Result:** liq 0.357 → 0.253, pro_gap 0.311. Better, but still not safe.
- **Decision:** USE_DOUBLE_DQN = True, USE_HUBER_LOSS = False, GRAD_CLIP_NORM = 10 (kept to the end).

**C4. Day-90 fix + 3000 episodes (v4)**
- **Problem 1 (bug):** the last day of an episode (day 90) was treated like "the world ends", so the agent learned wrong values near the end.
- **Fix:** only a real end (`terminated`) stops future value, not the time limit (`truncated`) (Pardo et al. 2018). ✅ kept.
- **Idea 2:** "maybe the agent is under-trained" → 3000 episodes instead of 1000.
- **Result:** liq 0.263, ctrl −0.33, high_risk 0.874. **No gain, even worse.**
- **Decision:** under-training rejected. Longer training is not the answer.

**C5. Danger-zone penalty (buf1)**
- **Idea:** charge a small penalty every day HF < 1.15, so sitting near the edge costs something.
- **Result:** liq **0.550** (worse); danger days did not drop.
- **Issue found:** the agent did not connect HF to value at all. A bigger stick does not help if it cannot see what the stick is for.
- **Decision:** penalty kept as an option but OFF. This also showed the problem is not the reward.

**Status after Phase C:** no agent was safe and earning. All failed in different ways. We needed to look inside the agent.

---

## 7. Phase D: diagnosis (29–30 Oct)

**D1. diagnose.py (Checks 1–3)**
- **Check 1:** compare the agent's predicted value with the real outcome. **Check 2:** what it chooses per HF zone. **Check 3:** HF sweep (same position, only debt changes).
- **Result:** predicted values far too high (v4: +62 predicted vs −4 real; v2: +144 vs +3). Same on TRAIN → not the market. Penalty changed nothing → not the reward. In every run the danger zone (HF < 1.1) was rated as good as the safe zone.
- **But:** choices inside a situation were sensible (repay when HF low, borrow when HF high).
- **First explanation (later corrected):** value over-estimation that "snowballs" through bootstrapping.

**D2. Check 0: the same check on every saved checkpoint (v4, ep100 → ep3000)**
- **First result (ep100, 300, 500, 1000, 2000, 3000):** safe at ep300–500 (liq 0.04–0.08), worse later (0.28–0.30).
- **First (wrong) reading:** "learned, then unlearned" because of the snowball; then "the replay buffer deletes old crash memories" (buffer 50k steps ≈ 555 episodes).
- **Second result (ep400 → ep1000, every 100):** 0.54, 0.08, 0.08, **0.72**, 0.40, 0.40, 0.28.
- **Corrected reading:** the agent did not learn then forget. It **swings** between safe and unsafe from checkpoint to checkpoint. ep400 was already bad (before the buffer was full), so the buffer was not the main cause.
- **Key clue:** predicted value stayed flat (~47–49) while real outcomes jumped from +12 to −42. The agent's scores for different actions were very close, so each big update flipped its choice → **policy churn** (Schaul et al. 2022).
- **Suspect:** learning rate 1e-3 too high (big updates) + target network copied in one jump every 500 steps.

**D3. Evaluate the good v4 checkpoints on 300 episodes**
- **Result:** ep300 liq 0.073, ep500 liq 0.057, ctrl 5.17 / 5.80; no accidental hedge; less danger time than v4 final.
- **Issue:** good snapshots of a shaky agent, not a stable level. Still entered danger 3x more than the heuristic.
- **Decision:** do not rely on lucky checkpoints. Fix the instability.

---

## 8. Phase E: the fix (30 Oct)

**E1. Learning rate 1e-3 → 1e-4 (one change only)**
- **Why:** smaller updates → the agent's choice should stop flipping. (Atari DQN used 2.5e-4, Rainbow 6.25e-5; 1e-3 is high.)
- **Result (seed 42, VAL 300):** **liq 0.000**, risk_adj 781.8, pro_gap 0.236, high_risk 0.057 (lower than the heuristic), hf_lt_1.1 0.001.
- **Check 0:** liq 0.00–0.04 at 9 of 10 checkpoints (v4: 0.04–0.72). Over-guess 62 → ~12, correlation 0.79.
- **Check 3:** it now clearly sees HF: "borrow" value drops from 29 (HF 2.0) to 1.8 (HF 1.05); it switches to protection at **HF 1.3**.
- **Issue found:** earns less (ctrl 4.66): it over-protects on calm days (parks USDC as collateral at 2% instead of the wallet at 5%, borrows more at 3%). Acts almost every day (gas cost on a real chain).
- **Decision:** LR 1e-4 kept. Check other seeds.

**E2. LR 1e-4 on seeds 1 and 2**
- **Result (VAL, ep1000):** 0/900 liquidations across 3 seeds; risk_adj 781.7; ctrl 4.52 ± 1.87; pro_gap 0.10 ± 0.12.
- **Check 0:** swings much smaller (mean ep500–1000: 0.33 → about 0.05), but seeds 1 and 2 still jump up to 0.18–0.20 at some checkpoints.
- **Issues:** proactive only clear for 1 of 3 seeds; earnings vary (2.58 to 6.32); seed 1 over-borrows ($1251 debt).
- **Decision:** try to remove the remaining jumps before the ablation (the ablation needs a fixed setup).

**E3. Soft target updates (tau 0.005), seed 1**
- **Why:** the target network was copied in one jump every 500 steps; a soft update moves it a little every step.
- **Result:** better early (ep200–500 all 0), but still 0.28 / 0.16 at ep600–700. Mean about the same (0.069 vs 0.077).
- **Decision (rule set before the run):** no clear gain → **setup frozen**: LR 1e-4, Double DQN, no Huber, clip 10, hard target copy (USE_SOFT_TARGET = False).
- **Lead (not proven):** with LR 1e-4 the bad checkpoints cluster at ep600–800, exactly when the replay buffer starts deleting the early exploration crashes. Possible future work.

---

## 9. Phase F: ablation study (30 Oct)

**What:** 4 groups × 3 seeds (42, 1, 2), same frozen setup, 1000 episodes each, evaluated on the same 300 VAL paths.

| group | R | CS | liq_rate | risk_adj | ctrl_gain | pro_gap | high_risk |
|---|---|---|---|---|---|---|---|
| static | | | 0.030 | 775.7 | 6.48 | 0 | 0.076 |
| heuristic | | | 0.000 | 784.3 | 7.23 | 0.104 | 0.097 |
| full | ✅ | ✅ | 0.000 | 781.7 ± 1.9 | 4.52 ± 1.87 | 0.10 ± 0.12 | 0.157 |
| r_only | ✅ | ❌ | 0.000 | 783.4 ± 0.2 | 6.28 ± 0.17 | 0.27 ± 0.24 | 0.248 |
| cs_only | ❌ | ✅ | 0.072 ± 0.059 | 759.2 ± 19.3 | 6.15 ± 1.30 | 0.44 ± 0.20 | 0.381 |
| **none** | ❌ | ❌ | **0.000** | **784.1 ± 0.5** | **6.99 ± 0.51** | 0.24 ± 0.16 | 0.117 |

**Findings**
1. Every group except cs_only: **0 liquidations** (0/2700 episodes).
2. **"none" is the best and most stable:** risk_adj equal to the heuristic, ctrl above static, pro_gap 2x the heuristic.
3. **Adding CS hurt every time** (r_only → full: ctrl 6.28 → 4.52; none → cs_only: liq 0 → 0.072).
4. **R did not make the agent more proactive.** Agents without R react to risk just as much. Likely reason: **HF is already in the observation**, and R is mostly built from HF.
5. R may act as a safety net when CS is present (cs_only → full: liq 0.072 → 0).
6. Only 3 seeds: most differences are trends, not statistically proven.

**Open question (next step):** what do `--no-r` / `--no-cs` actually remove? Only the observation, or also the reward (CS) and the replay priority (PER boosts R > 0.7)? This decides how to explain the findings.

---

## 10. Ideas considered but NOT run (and why)

| idea | why not (yet) |
|---|---|
| n-step returns (Rainbow) | planned for the over-guess, but the over-guess turned out to be mostly instability, fixed by LR |
| Bigger replay buffer | buffer was ruled out as the main cause; still a possible future fix for the ep600–800 dips |
| Stress training data (crash paths) | not needed once agents were safe; useful future work for falling markets |
| Safety shield (if HF < 1.1 always repay) | fallback for Sepolia; not needed since agents are safe |
| Early stopping (pick best checkpoint) | rejected: the good checkpoints were lucky snapshots; final checkpoint used for every seed instead |
| Small action cost (gas) | future work; current agents act almost daily |

---

## 11. Wrong guesses we corrected (be honest about these)

| what we first thought | what the evidence showed |
|---|---|
| v2 is safe because it understands risk | safe by an accidental ETH-borrowing hedge |
| more training helps | 3000 episodes was worse than 1000 |
| a reward penalty will teach caution | it made things worse; the problem was learning, not reward |
| over-estimation "snowball" is the root cause | the guesses were flat; the policy was swinging (instability) |
| "learned at ep300–500, then unlearned" | it was swinging all along (ep400 = 0.54) |
| the replay buffer deletes the lessons | ep400 was bad before the buffer was full; not the main cause |
| the target-network jump causes the swings | soft updates did not remove them |
| R makes the agent proactive | agents without R are just as proactive (HF is visible) |

---

## 12. What you can claim now

**Supported ✅**
- A DQN agent can learn to keep a DeFi loan safe: 0 liquidations in 2700 VAL episodes (9 runs), on a falling market it never trained on.
- It is proactive: it protects at about HF 1.3 and reacts to risk 2x more than a hand-made rule (pro_gap).
- The best agents match the heuristic on risk-adjusted return.
- **Method finding:** the main reason DQN failed was a too-high learning rate (policy churn), not the reward or the market. Shown with a diagnosis tool, and many alternative explanations were tested and ruled out.

**Not supported / limitations ❌**
- "R in the state makes the agent proactive" (needs the open question answered).
- CS in the state lowers earnings and adds variance (may be partly by design: losses weigh more for low CS).
- Agents earn about the same or less than the heuristic; none clearly beats it.
- Checkpoints still swing a little (up to about 0.2 on 50 episodes).
- LR chosen on VAL → final numbers must come from TEST.
- 3 seeds only; 50-episode checks are noisy; simulator has fixed rates and no gas.

---

## 13. Next steps

1. Check what `--no-r` / `--no-cs` remove (`grep` in lending_env.py and training_loop.py).
2. Decide the framing of R and CS (and whether extra seeds are needed).
3. TEST evaluation of all 12 runs, **once**.
4. Choose the agent to deploy (fixed rule) and write `policy.py` for the friend (Sepolia).
