"""Small deterministic query agent for common AML investigation questions."""

import re
from datetime import timedelta

import pandas as pd

from src.tools.anomaly_tool import detect_structuring
from src.tools.explainer_tool import explain_risk
from src.tools.feature_tool import add_aml_features
from src.tools.filter_tool import filter_transactions


def _find_customer_id(query: str) -> str | None:
    match = re.search(r"\bCUST_\d{4}(?:_[A-Z]+)?\b", query.upper())
    return match.group(0) if match else None


def _amount_from_query(query: str) -> float | None:
    match = re.search(r"(?:under|below|less than)\s+\$?([\d,]+(?:\.\d+)?)", query.lower())
    return float(match.group(1).replace(",", "")) if match else None


def _days_from_query(query: str) -> int | None:
    match = re.search(r"last\s+(\d+)\s+days?", query.lower())
    return int(match.group(1)) if match else None


def answer_query(transactions: pd.DataFrame, query: str) -> dict[str, object]:
    """Answer a supported AML question with an explanation and supporting rows.

    Supported intents include customer-risk checks, structuring searches, and
    transaction searches such as "under $10,000". The returned dictionary is
    deliberately UI-agnostic so it can also be used by an API later.
    """
    normalized_query = query.strip()
    if not normalized_query:
        return {"title": "Ask an AML question", "answer": "Enter a question to begin.", "data": pd.DataFrame()}

    customer_id = _find_customer_id(normalized_query)
    if customer_id:
        customer_data = filter_transactions(transactions, customer_id=customer_id)
        if customer_data.empty:
            return {
                "title": "Customer not found",
                "answer": f"No transactions were found for {customer_id}.",
                "data": customer_data,
            }

        structuring = detect_structuring(customer_data)
        if not structuring.empty:
            alert = explain_risk(structuring)[0]
            return {"title": "High-risk customer", "answer": alert["explanation"], "data": structuring, "action": alert["recommended_action"]}

        features = add_aml_features(customer_data)
        velocity = features[(features["txn_count_24h"] >= 10) & features["is_high_risk_country"]]
        if not velocity.empty:
            return {
                "title": "High-risk customer",
                "answer": (
                    f"{customer_id} made {len(customer_data)} transactions, including "
                    f"{len(velocity)} rapid transactions in a high-risk jurisdiction. "
                    "Recommend flagging this customer for review."
                ),
                "data": velocity,
                "action": "FLAG FOR REVIEW",
            }
        return {
            "title": "Customer review",
            "answer": f"No rule-based structuring or high-velocity alert was found for {customer_id}.",
            "data": customer_data,
            "action": "NO IMMEDIATE ACTION",
        }

    if "structur" in normalized_query.lower():
        scoped_data = transactions
        days = _days_from_query(normalized_query)
        if days is not None:
            reference_date = transactions["timestamp"].max()
            scoped_data = filter_transactions(transactions, date_start=reference_date - timedelta(days=days))
        structuring = detect_structuring(scoped_data)
        if structuring.empty:
            return {"title": "Structuring search", "answer": "No structuring patterns were found in the requested period.", "data": structuring}
        customers = ", ".join(structuring["customer_id"].unique())
        return {
            "title": "Structuring search",
            "answer": f"Found {len(structuring)} linked transactions associated with: {customers}.",
            "data": structuring,
            "action": "FILE SAR REPORT",
        }

    amount = _amount_from_query(normalized_query)
    if amount is not None:
        matches = filter_transactions(transactions, max_amount=amount)
        customer_counts = (
            matches.groupby("customer_id", as_index=False)
            .agg(transaction_count=("transaction_id", "size"), total_amount=("amount", "sum"))
            .sort_values(["transaction_count", "total_amount"], ascending=False)
        )
        return {
            "title": "Transaction search",
            "answer": f"Found {len(matches):,} transactions at or below {amount:,.2f} across {len(customer_counts):,} customers.",
            "data": customer_counts,
        }

    return {
        "title": "Supported questions",
        "answer": (
            "Try: 'Is CUST_9999_STRUCTURER suspicious?', "
            "'Find structuring patterns in the last 60 days', or "
            "'Which customers made transactions under $10,000?'"
        ),
        "data": pd.DataFrame(),
    }
