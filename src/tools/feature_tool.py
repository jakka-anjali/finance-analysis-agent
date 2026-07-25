"""On-demand AML feature engineering."""

import pandas as pd


def add_aml_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add per-transaction rolling velocity and structuring features."""
    result = transactions.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result = result.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    result["is_near_threshold"] = result["amount"].between(9000, 9999.99, inclusive="both")
    result["is_high_risk_country"] = result["country"].isin(["KY", "PA"])

    chunks: list[pd.DataFrame] = []
    for _, customer_transactions in result.groupby("customer_id", sort=False):
        customer_transactions = customer_transactions.copy().set_index("timestamp")
        customer_transactions["txn_count_24h"] = customer_transactions["amount"].rolling("24h").count().astype(int)
        customer_transactions["txn_count_48h"] = customer_transactions["amount"].rolling("48h").count().astype(int)
        customer_transactions["amount_sum_24h"] = customer_transactions["amount"].rolling("24h").sum()
        chunks.append(customer_transactions.reset_index())
    return pd.concat(chunks, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def customer_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return a customer-level summary including a near-threshold ratio."""
    source = transactions.copy()
    source["is_near_threshold"] = source["amount"].between(9000, 9999.99, inclusive="both")
    summary = source.groupby("customer_id").agg(
        transaction_count=("transaction_id", "size"),
        total_amount=("amount", "sum"),
        near_threshold_count=("is_near_threshold", "sum"),
        high_risk_country_count=("country", lambda x: x.isin(["KY", "PA"]).sum()),
    )
    summary["near_threshold_ratio"] = summary["near_threshold_count"] / summary["transaction_count"]
    return summary.reset_index()
