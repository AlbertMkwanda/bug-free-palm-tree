"""Deterministic synthetic mobile-money transaction data for demonstration."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "mobile_money_transactions.csv"

FEATURES = [
    "amount_mwk",
    "transaction_type",
    "channel",
    "hour",
    "new_device",
    "new_recipient",
    "location_change",
    "transactions_10m",
    "account_age_days",
    "kyc_tier",
]
CATEGORICAL_FEATURES = ["transaction_type", "channel", "kyc_tier"]
NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES]


def generate_dataset(rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Create a labeled, synthetic Malawi-style mobile-money dataset."""
    rng = np.random.default_rng(seed)
    transaction_type = rng.choice(
        ["person_to_person", "merchant_payment", "cash_out", "airtime"],
        rows,
        p=[0.46, 0.27, 0.19, 0.08],
    )
    channel = rng.choice(["USSD", "mobile_app", "agent"], rows, p=[0.54, 0.31, 0.15])
    kyc_tier = rng.choice(["basic", "enhanced"], rows, p=[0.62, 0.38])
    amount = np.clip(rng.lognormal(9.15, 1.05, rows), 200, 5_000_000).round(2)
    amount[transaction_type == "airtime"] = np.clip(
        rng.lognormal(7.0, 0.65, (transaction_type == "airtime").sum()), 100, 250_000
    ).round(2)
    hour = rng.integers(0, 24, rows)
    new_device = rng.binomial(1, 0.055, rows)
    new_recipient = rng.binomial(1, 0.12, rows)
    location_change = rng.binomial(1, 0.065, rows)
    transactions_10m = rng.poisson(0.45, rows)
    account_age_days = np.maximum(1, rng.gamma(4.0, 210, rows).astype(int))

    risk_logit = (
        -4.0
        + 1.8 * (amount > 500_000)
        + 1.7 * new_device
        + 1.9 * new_recipient
        + 1.5 * location_change
        + 1.35 * (transactions_10m >= 3)
        + 0.75 * (transaction_type == "cash_out")
        + 0.65 * ((hour <= 4) | (hour >= 23))
        + 1.2 * (account_age_days < 30)
    )
    fraud_probability = 1 / (1 + np.exp(-risk_logit))
    is_fraud = rng.binomial(1, fraud_probability)

    frame = pd.DataFrame(
        {
            "transaction_id": [f"MM{index:07d}" for index in range(1, rows + 1)],
            "account_id": [f"AC{value:05d}" for value in rng.integers(1, 1601, rows)],
            "amount_mwk": amount,
            "transaction_type": transaction_type,
            "channel": channel,
            "hour": hour,
            "new_device": new_device,
            "new_recipient": new_recipient,
            "location_change": location_change,
            "transactions_10m": transactions_10m,
            "account_age_days": account_age_days,
            "kyc_tier": kyc_tier,
            "is_fraud": is_fraud,
        }
    )
    return frame


def load_or_generate(path: Path = DATA_PATH, force: bool = False) -> pd.DataFrame:
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset().to_csv(path, index=False)
    return pd.read_csv(path)