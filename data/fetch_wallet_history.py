"""
fetch_wallet_history.py

Pulls real wallet-level lending history from Aave V3 (Ethereum mainnet)
via The Graph, to compute Credit Score (CS) features:
  - repayment rate
  - historical LTV trajectory / borrow-vs-repay behavior
  - past liquidation count
  - account activity span (proxy for "age")

Why mainnet Aave instead of Sepolia testnet:
  Testnet wallets are mostly bots/devs with no genuine financial behavior,
  so there's nothing meaningful to learn credit patterns from. Real Aave
  mainnet data gives real repayment/liquidation behavior.

Subgraph used:
  Aave's own official V3 Ethereum Mainnet subgraph (not a third-party
  guess) -- schema and query shape confirmed directly from Aave's
  protocol-subgraphs repo docs: https://github.com/aave/protocol-subgraphs
  Subgraph ID: Cd2gEDVeqnjBn1hSeqFMitw8Q1iiyV9FYUZkLNRcL87g

Usage:
    python fetch_wallet_history.py

Requirements:
  - Free API key from https://thegraph.com/studio/apikeys/ (first 100,000
    queries/month are free, no card required). Set THEGRAPH_API_KEY.

Notes on approach:
  - The `userTransactions` entity uses a GraphQL union type for `action`,
    with inline fragments per transaction type (Supply, Borrow, Repay,
    LiquidationCall, etc.) -- this is Aave's actual documented schema,
    not guessed.
  - Pagination cap is 100 items per query (The Graph's standard limit),
    handled here via first/skip.
  - We first pull a page of distinct users via the `users` entity, then
    fetch each user's transaction history. If `users` isn't queryable
    this way on this subgraph version, check the Explorer Playground's
    schema tab (left sidebar) for the correct top-level list field --
    it may be `userReserves` grouped by user instead.
"""

import os
import time
import requests
import pandas as pd

# ---- Config ----
API_KEY = os.environ.get("THEGRAPH_API_KEY", "")
SUBGRAPH_ID = "Cd2gEDVeqnjBn1hSeqFMitw8Q1iiyV9FYUZkLNRcL87g"  # Aave V3 Ethereum Mainnet (official)
GRAPHQL_URL = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw")
NUM_WALLETS = 500
PAGE_SIZE = 100  # The Graph's per-query item cap


USERS_QUERY = """
query GetUsers($first: Int!, $skip: Int!) {
  users(first: $first, skip: $skip) {
    id
  }
}
"""

USER_TRANSACTIONS_QUERY = """
query GetUserTransactions($user: String!, $first: Int!, $skip: Int!) {
  userTransactions(
    where: { user: $user }
    orderBy: timestamp
    orderDirection: asc
    first: $first
    skip: $skip
  ) {
    id
    timestamp
    action
    ... on Supply {
      amount
      assetPriceUSD
      reserve { symbol decimals }
    }
    ... on Borrow {
      amount
      assetPriceUSD
      reserve { symbol decimals }
    }
    ... on Repay {
      amount
      assetPriceUSD
      reserve { symbol decimals }
    }
    ... on LiquidationCall {
      collateralAmount
      principalAmount
      collateralAssetPriceUSD
      borrowAssetPriceUSD
      collateralReserve { symbol decimals }
      principalReserve { symbol decimals }
    }
  }
}
"""


def run_query(query: str, variables: dict) -> dict:
    resp = requests.post(
        GRAPHQL_URL,
        json={"query": query, "variables": variables},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "errors" in payload:
        raise RuntimeError(f"GraphQL errors: {payload['errors']}")
    return payload["data"]


BURN_ADDRESSES = {
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
}


def fetch_wallet_addresses(num_wallets: int, page_size: int) -> list[str]:
    addresses = []
    skip = 0
    while len(addresses) < num_wallets:
        data = run_query(USERS_QUERY, {"first": page_size, "skip": skip})
        batch = data["users"]
        if not batch:
            break
        addresses.extend(
            u["id"] for u in batch
            if u["id"].lower() not in BURN_ADDRESSES
        )
        skip += page_size
        time.sleep(0.2)
    return addresses[:num_wallets]


def fetch_all_transactions_for_user(address: str) -> list[dict]:
    """Paginate through one user's full transaction history (up to a few pages)."""
    all_txns = []
    skip = 0
    while True:
        data = run_query(USER_TRANSACTIONS_QUERY, {
            "user": address, "first": PAGE_SIZE, "skip": skip,
        })
        batch = data["userTransactions"]
        all_txns.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        skip += PAGE_SIZE
        if skip >= 1000:  # sanity cap per wallet -- adjust if needed
            break
        time.sleep(0.1)
    return all_txns


def _usd_amount(t: dict) -> float:
    """Convert a raw on-chain amount to real USD value: normalize by the
    reserve's decimals, then multiply by the price at time of transaction.
    This makes amounts from different reserves (WETH, USDC, DAI, ...)
    directly comparable and summable."""
    decimals = t.get("reserve", {}).get("decimals", 18)
    price_usd = float(t.get("assetPriceUSD", 0) or 0)
    token_amount = float(t.get("amount", 0)) / (10 ** int(decimals))
    return token_amount * price_usd


def compute_credit_features(address: str, txns: list[dict]) -> dict:
    """Turn raw transaction history into CS input features, with real
    USD-valued totals (decimals-normalized amount x price at tx time)."""
    supplies = [t for t in txns if t.get("action") == "Supply"]
    borrows = [t for t in txns if t.get("action") == "Borrow"]
    repays = [t for t in txns if t.get("action") == "Repay"]
    liquidations = [t for t in txns if t.get("action") == "LiquidationCall"]

    total_supplied_usd = sum(_usd_amount(t) for t in supplies)
    total_borrowed_usd = sum(_usd_amount(t) for t in borrows)
    total_repaid_usd = sum(_usd_amount(t) for t in repays)

    repayment_rate = (total_repaid_usd / total_borrowed_usd) if total_borrowed_usd > 0 else None
    # True LTV: how much was borrowed relative to how much was supplied, in real dollars.
    ltv = (total_borrowed_usd / total_supplied_usd) if total_supplied_usd > 0 else None

    timestamps = [int(t["timestamp"]) for t in txns]
    account_age_days = (
        (max(timestamps) - min(timestamps)) / 86400 if timestamps else None
    )

    return {
        "wallet": address,
        "num_supplies": len(supplies),
        "num_borrows": len(borrows),
        "num_repays": len(repays),
        "total_supplied_usd": total_supplied_usd,
        "total_borrowed_usd": total_borrowed_usd,
        "total_repaid_usd": total_repaid_usd,
        "repayment_rate": repayment_rate,
        "ltv": ltv,
        "num_liquidations": len(liquidations),
        "account_age_days": account_age_days,
        "total_transactions": len(txns),
    }


def main():
    if not API_KEY:
        print("WARNING: THEGRAPH_API_KEY not set. Requests will fail.")
        print("Get a free key at https://thegraph.com/studio/apikeys/")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Fetching up to {NUM_WALLETS} wallet addresses...")
    addresses = fetch_wallet_addresses(NUM_WALLETS, PAGE_SIZE)
    print(f"  -> got {len(addresses)} addresses")

    features = []
    for i, addr in enumerate(addresses, 1):
        txns = fetch_all_transactions_for_user(addr)
        features.append(compute_credit_features(addr, txns))
        if i % 25 == 0:
            print(f"  processed {i}/{len(addresses)} wallets...")
        time.sleep(0.1)

    df = pd.DataFrame(features)
    out_path = os.path.join(OUTPUT_DIR, "wallet_credit_features.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} wallets to {out_path}")
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()