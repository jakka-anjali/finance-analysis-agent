"""Risk classification for flagged transactions and customers."""

from __future__ import annotations

import pandas as pd


def classify_risk(
    flagged_transactions: pd.DataFrame,
    *,
    intent: str = "general_aml_analysis",
) -> list[dict[str, object]]:
    """Convert detection signals into low / medium / high risk categories."""
    if flagged_transactions.empty:
        return []

    data = flagged_transactions.copy()
    alerts: list[dict[str, object]] = []

    for customer_id, group in data.groupby("customer_id"):
        count = len(group)
        total = float(group["amount"].sum())
        has_structuring = "rule_flag" in group.columns and (group["rule_flag"] == "STRUCTURING").any()
        has_ml = "ml_flag" in group.columns and (group["ml_flag"] == "ML_ANOMALY").any()
        high_velocity = "txn_count_24h" in group.columns and (group["txn_count_24h"] >= 10).any()
        high_risk_country = "is_high_risk_country" in group.columns and group["is_high_risk_country"].any()

        score = 0
        signals: list[str] = []

        if has_structuring:
            score += 60
            signals.append("structuring pattern")
        if has_ml:
            score += 35
            signals.append("ML anomaly score")
        if high_velocity:
            score += 25
            signals.append("high transaction velocity")
        if high_risk_country:
            score += 15
            signals.append("high-risk jurisdiction exposure")
        if count >= 10:
            score += 10
            signals.append("elevated transaction count")

        if score >= 60:
            risk_level = "High"
            escalation = "REPORT"
        elif score >= 30:
            risk_level = "Medium"
            escalation = "REVIEW"
        else:
            risk_level = "Low"
            escalation = "MONITOR"

        if intent == "aggregation_query" and count >= 10:
            risk_level = "Medium" if count < 20 else "High"
            escalation = "REVIEW" if risk_level == "Medium" else "REPORT"
            signals.append(f"{count} qualifying transactions")

        alerts.append(
            {
                "customer_id": customer_id,
                "risk_level": risk_level,
                "risk_score": min(score, 100),
                "signals": signals,
                "transaction_count": count,
                "total_amount": total,
                "escalation": escalation,
            }
        )

    return sorted(alerts, key=lambda item: item["risk_score"], reverse=True)
