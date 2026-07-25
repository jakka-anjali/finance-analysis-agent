"""Customer-level aggregation for threshold-based AML queries."""

from __future__ import annotations

import pandas as pd


def aggregate_customers(
    transactions: pd.DataFrame,
    *,
    max_amount: float | None = None,
    min_amount: float | None = None,
    min_transaction_count: int = 1,
) -> pd.DataFrame:
    """Roll up transactions by customer and apply count / amount thresholds."""
    if transactions.empty:
        return pd.DataFrame()

    scoped = transactions.copy()
    if max_amount is not None:
        scoped = scoped[scoped["amount"] <= max_amount]
    if min_amount is not None:
        scoped = scoped[scoped["amount"] >= min_amount]

    summary = (
        scoped.groupby("customer_id", as_index=False)
        .agg(
            transaction_count=("transaction_id", "size"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
            max_amount=("amount", "max"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
        )
        .sort_values(["transaction_count", "total_amount"], ascending=False)
    )

    if min_transaction_count > 1:
        summary = summary[summary["transaction_count"] >= min_transaction_count]

    return summary.reset_index(drop=True)
