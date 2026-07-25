"""Generate analyst-ready AML alert explanations and escalation guidance."""

from __future__ import annotations

import pandas as pd

from src.tools.risk_tool import classify_risk

ESCALATION_PLAYBOOK = {
    "REPORT": {
        "action": "File SAR / escalate to MLRO",
        "timeline": "Within 24–48 hours",
        "steps": [
            "Preserve all related transaction records and audit logs.",
            "Document the structuring or smurfing indicators in the case narrative.",
            "Cross-check linked accounts, beneficiaries, and counterparties.",
            "Submit SAR filing after MLRO review and approval.",
        ],
    },
    "REVIEW": {
        "action": "Flag for enhanced due diligence (EDD)",
        "timeline": "Within 5 business days",
        "steps": [
            "Assign the case to a senior analyst for manual review.",
            "Request source-of-funds documentation from the relationship manager.",
            "Compare activity against the customer's expected profile and KYC data.",
            "Decide whether to maintain monitoring, restrict activity, or escalate to SAR.",
        ],
    },
    "MONITOR": {
        "action": "Add to watchlist and continue monitoring",
        "timeline": "Ongoing — re-evaluate in 30 days",
        "steps": [
            "Place the customer on the transaction monitoring watchlist.",
            "Set alert thresholds for repeat near-threshold or velocity patterns.",
            "Review again if activity volume or jurisdiction mix changes materially.",
        ],
    },
}


def _analyst_advice(risk_level: str, escalation: str, signals: list[str]) -> str:
    signal_text = ", ".join(signals) if signals else "unusual activity indicators"
    playbook = ESCALATION_PLAYBOOK.get(escalation, ESCALATION_PLAYBOOK["MONITOR"])

    return (
        f"**Analyst recommendation:** {playbook['action']} ({playbook['timeline']}). "
        f"The {risk_level.lower()}-risk rating is driven by {signal_text}. "
        "Next steps: "
        + " → ".join(playbook["steps"][:3])
        + "."
    )


def explain_risk(
    flagged_transactions: pd.DataFrame,
    *,
    intent: str = "general_aml_analysis",
    original_query: str = "",
) -> list[dict[str, object]]:
    """Summarise alerts by customer with explanations and analyst guidance."""
    if flagged_transactions.empty:
        return []

    classifications = classify_risk(flagged_transactions, intent=intent)
    data = flagged_transactions.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    alerts: list[dict[str, object]] = []

    for classification in classifications:
        customer_id = classification["customer_id"]
        group = data[data["customer_id"] == customer_id]
        count = len(group)
        total = float(group["amount"].sum())
        first_seen = group["timestamp"].min().strftime("%Y-%m-%d %H:%M")
        last_seen = group["timestamp"].max().strftime("%Y-%m-%d %H:%M")
        risk_level = classification["risk_level"]
        escalation = classification["escalation"]
        signals = classification["signals"]

        has_structuring = "rule_flag" in group.columns and (group["rule_flag"] == "STRUCTURING").any()
        if has_structuring:
            explanation = (
                f"Customer **{customer_id}** triggered a structuring alert: {count} cash deposits "
                f"between $9,000 and $9,999 from {first_seen} to {last_seen} "
                f"(total ${total:,.2f}). This pattern is consistent with deliberate threshold evasion."
            )
        elif "ml_flag" in group.columns and (group["ml_flag"] == "ML_ANOMALY").any():
            top_score = float(group["anomaly_score"].max()) if "anomaly_score" in group.columns else 0.0
            explanation = (
                f"Customer **{customer_id}** has {count} ML-flagged outlier transactions "
                f"from {first_seen} to {last_seen} (peak anomaly score {top_score:.3f}). "
                "Behaviour deviates from baseline transaction patterns."
            )
        else:
            explanation = (
                f"Customer **{customer_id}** has {count} flagged transactions totalling "
                f"${total:,.2f} from {first_seen} to {last_seen}."
            )

        if original_query:
            explanation += f" This assessment addresses your query: \"{original_query}\"."

        recommended_action = ESCALATION_PLAYBOOK[escalation]["action"]
        analyst_advice = _analyst_advice(risk_level, escalation, signals)

        alerts.append(
            {
                "customer_id": customer_id,
                "risk_level": risk_level,
                "risk_score": classification["risk_score"],
                "signals": signals,
                "explanation": explanation,
                "recommended_action": recommended_action,
                "escalation": escalation,
                "analyst_advice": analyst_advice,
                "playbook_steps": ESCALATION_PLAYBOOK[escalation]["steps"],
            }
        )

    return alerts
