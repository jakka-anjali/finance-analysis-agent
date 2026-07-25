
"""Exploratory Data Analysis tool for AML data profiling."""

import pandas as pd


def run_eda(transactions: pd.DataFrame) -> dict[str, object]:
    """Perform dataset profiling and return key baseline metrics."""
    if transactions.empty:
        return {"error": "No transaction data available for analysis."}

    data = transactions.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    return {
        "total_transactions": int(len(data)),
        "unique_customers": int(data["customer_id"].nunique()),
        "date_range": {
            "min": data["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "max": data["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "amount_stats": {
            "mean": float(round(data["amount"].mean(), 2)),
            "std": float(round(data["amount"].std(), 2)),
            "min": float(round(data["amount"].min(), 2)),
            "median": float(round(data["amount"].median(), 2)),
            "max": float(round(data["amount"].max(), 2)),
        },
        "channels_distribution": data["channel"].value_counts().to_dict(),
        "high_risk_jurisdictions_count": int(
            data["country"].isin(["KY", "PA"]).sum()
        ),
        "label_distribution": data["ground_truth_label"].value_counts().to_dict(),
    }