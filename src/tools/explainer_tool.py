"""Generate clear, structured AML alert explanations."""

import pandas as pd


def explain_risk(flagged_transactions: pd.DataFrame) -> list[dict[str, object]]:
    """Summarise alerts by customer in JSON-ready dictionaries."""
    if flagged_transactions.empty:
        return []

    data = flagged_transactions.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    alerts: list[dict[str, object]] = []
    for customer_id, group in data.groupby("customer_id"):
        count = len(group)
        total = float(group["amount"].sum())
        first_seen = group["timestamp"].min().strftime("%Y-%m-%d %H:%M")
        last_seen = group["timestamp"].max().strftime("%Y-%m-%d %H:%M")
        has_structuring = "rule_flag" in group and (group["rule_flag"] == "STRUCTURING").any()
        if has_structuring:
            risk_level, action = "High", "FILE SAR REPORT"
            explanation = (
                f"Customer {customer_id} made {count} cash deposits between $9,000 and $9,999 "
                f"from {first_seen} to {last_seen}, consistent with potential structuring."
            )
        else:
            risk_level = "Medium" if count >= 3 else "Low"
            action = "FLAG FOR REVIEW"
            explanation = (
                f"Customer {customer_id} has {count} unusual transactions totalling ${total:,.2f} "
                f"from {first_seen} to {last_seen}."
            )
        alerts.append(
            {
                "customer_id": customer_id,
                "risk_level": risk_level,
                "explanation": explanation,
                "recommended_action": action,
            }
        )
    return alerts
