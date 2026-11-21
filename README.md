# auto_defi: a credit-aware DQN agent for DeFi lending, with an R safety shield

This repo is the **left part** of the auto_defi thesis: data, the Risk Signal (R), the Credit Score (CS), the lending simulator, and the offline training of a DQN agent.
The **right part** (the live service on Sepolia) uses this repo through **one file: `policy.py`**.

If you only want to connect the agent to the live app, read sections 1–5. Sections 6–8 explain why it works this way and where the full story is.

---

## 1. What the agent does (in one minute)

Every day the agent looks at one user's loan and picks **one action**:

| id | action | id | action |
|---|---|---|---|
| 0 | deposit_ETH | 4 | deposit_USDC |
| 1 | withdraw_ETH | 5 | withdraw_USDC |
| 2 | borrow_ETH | 6 | borrow_USDC |
| 3 | repay_ETH | 7 | repay_USDC |
| 8 | **hold** (do nothing) | | |

Every action moves **$100** worth of the asset (less if the wallet, collateral or debt is smaller).

It uses two extra numbers that a normal lending bot does not have:

- **CS (Credit Score, 0–1):** from the user's on-chain history. It sets the user's borrow rate: `2% + 6% × (1 − CS)`.
  A good user (CS 0.95) pays 2.3% and borrowing is worth it; a risky user (CS 0.25) pays 6.5% and should repay instead.
  *Example: a bank gives a cheap loan to a customer with a good history.*
- **R (Risk Signal):** the chance that the loan gets liquidated in the next 5 days, from the Health Factor (HF) and today's market volatility.
  *Example: HF says how far the cliff edge is; R also says how fast you are driving.*

**Deployment agent:** `cs_only_seed42_env4` (it sees CS, not R). It had **0 liquidations** on VAL, STRESS and the final TEST.

---

## 2. The safety shield (why the agent is not used "alone")

On top of the agent sits a simple rule driven by R, the **shield with memory**:

- **ON** when R > 5% → borrow and withdraw are **blocked** (the agent picks its best safe action instead).
- **OFF** only when R ≤ 1% **and** at least 5 days have passed since R was last above 5%.

*Example: after a storm warning a parent keeps the car keys until the sky is really clear AND the warning is 5 days old, not the second the rain stops.*

Why: the experiments showed that agents react to danger but then **undo** their own safety moves (repay, then borrow again). The shield stops that.
Result: in hard (STRESS) markets it cut liquidations by about **30–50%** (two different test seeds), and in normal markets it **costs nothing** (largest money drop $0.03 on the final TEST).

The shield is already **inside `policy.py`**. You do not need to add anything.

---

## 3. Files the live service needs

```
policy.py                                  <- the only file the service calls
dqn_agent.py                               <- network shape (QNetwork)
features/compute_risk_signal.py            <- the R formula
checkpoints/cs_only_seed42_env4/final.pt   <- the trained agent
data/raw/eth_daily.csv                     <- only needed if you do NOT call warm_up()
simulation/price_simulator.py              <- same as above
```

Python packages: `numpy`, `torch` (and `pandas` only for the `eth_daily.csv` fallback). See `requirements.txt`.

---

## 4. How to use it (copy this)

```python
from policy import LivePolicy

pol = LivePolicy()                    # loads cs_only_seed42_env4 with the shield ON

# once, at start: recent real daily ETH closing prices, oldest first (about 60 days)
pol.warm_up(last_60_daily_eth_closes)

# then ONCE PER DAY, with the new daily price:
position = {
    "wallet_ETH": 0.0,      "wallet_USDC": 400.0,    # TOKEN amounts, not USD
    "collat_ETH": 0.333,    "collat_USDC": 0.0,
    "debt_ETH": 0.0,        "debt_USDC": 400.0,
}
d = pol.decide(eth_price=3000.0, position=position, credit_score=0.72)

if d.action != 8:
    # send d.action_type ("repay") of d.amount tokens of d.asset ("USDC") to the pool
    ...
```

What you get back (`d`):

| field | meaning |
|---|---|
| `action` | 0–8 (8 = hold) |
| `action_name` | e.g. `"repay_USDC"` |
| `action_type`, `asset` | e.g. `"repay"`, `"USDC"` (None for hold) |
| `amount` | **token** amount to send (0 for hold) |
| `R` | today's liquidation risk (0–1) |
| `health_factor` | today's HF |
| `shield_on` | True if borrow/withdraw are blocked today |
| `changed_by_shield` | True if the agent wanted a risky move and the shield stopped it (good to log) |

---

## 5. Four rules (please read)

1. **Give token amounts, not USD.** `collat_ETH = 0.333` means 0.333 ETH. Use the exact key names above.
2. **Call `decide()` once per day.** Each call is "a new day": it updates the volatility and the shield's 5-day clock.
   Calling it 10 times a day would turn the shield off after a few hours instead of 5 days.
   To ask again on the same day (for example to show it in the UI), use `decide(..., new_day=False)`.
   *Example: one pill per day; ten pills in one day breaks the treatment.*
3. **Call `warm_up()` once at the start** with real recent daily closes, so R knows today's market. Without it, R starts from the average training volatility.
4. **Do not change the constants** at the top of `policy.py` ($100 action size, HF rule 1.05, shield 5% / 1% / 5 days). They must match the training; `test_policy.py` checks this.

One user = one `LivePolicy` object (it keeps that user's volatility, price and shield memory). For a new user or a new loan, create a new one or call `pol.reset()`.

---

## 6. How we know `policy.py` is right

```bash
python test_policy.py                                          # VAL, deployment agent
python test_policy.py --split stress --run full_seed42_env4    # stormy markets, shield busy
```

`test_policy.py` runs the simulator and, on every day, gives `policy.py` only what the live service has (token amounts, price, CS). It must pick **the same action** as `evaluate.py`, the script that produced all thesis results.

Result (21 Nov 2026): **5/5 checks passed**, 0 action mismatches in 4,500 steps each, also with the shield ON on 17.9% of days (STRESS, full_seed42).

---

## 7. Main results (short)

| Claim | Status |
|---|---|
| The agent keeps loans safe (TEST liq_rate ≤ ~1% for all learning groups) | ✅ |
| The agent is credit-aware: high-CS users get bigger loans, low-CS smaller (cs_only, 4 of 5 seeds) | ✅ |
| R is a correct early warning (it almost never misses a slow danger) | ✅ |
| R as an agent input alone makes it safer | ❌ (agents react, then undo) |
| R as a safety shield makes it safer (STRESS −31% to −48% liquidations) at no money cost | ✅ |
| Beats a strong hand-made rule on money | ❌ (heuristic_cs $5.30 vs cs_only $4.73) |
| One-day crashes (10%+) can be prevented | ❌ (no warning possible; 30–55% of remaining liquidations) |

Data splits: **VAL** = for decisions, **STRESS** = stormy markets with 3× more crashes (testing only), **TEST** = final, used once (seed 3031, closed).

---

## 8. The full story

| Document | What is inside |
|---|---|
| `docs/experiment_journey_part1.md` | up to 30 Oct: building a correct, fair simulator, why the first DQN agents failed (learning rate too high → the agent's choices kept flipping), the fix (LR 1e-4), first ablation |
| `docs/experiment_journey_part2.md` | 6–7 Nov: why R and CS first did not help, the new environment (CS sets the rate, R = 5-day liquidation chance, calm/stormy markets), 20-agent ablation, final TEST |
| `docs/experiment_journey_part3.md` | 18–21 Nov: action counts, an action fee (rejected), why R agents were not safer (they undo), the shield with memory, confirmation, final TEST with the shield |

Read them in order (1 → 2 → 3). Each journey file has a glossary, every result table, the "wrong guesses we corrected", and the exact commands.

---

## 9. Quick glossary

| Word | Meaning |
|---|---|
| HF (Health Factor) | How safe the loan is. Below 1.0 = liquidation. |
| Liquidation | The pool sells part of the collateral to repay the debt; the user loses money. |
| R | Chance of liquidation in the next 5 days. |
| CS | Credit Score 0–1 from on-chain history; sets the borrow rate. |
| Shield | Rule that blocks borrow/withdraw while R is high (with memory). |
| liq_rate | Share of 90-day episodes with at least one liquidation. |
| ctrl_gain | Money the strategy controls: interest earned − interest paid + liquidation loss ($ per episode). |
| full / r_only / cs_only / none | Agent groups: sees R and CS / only R / only CS / neither. |
