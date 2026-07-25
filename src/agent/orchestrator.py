"""Unified AML agent orchestrator — dynamic plan construction and tool execution."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from src.agent.intent_parser import ParsedQuery, parse_query
from src.tools.aggregation_tool import aggregate_customers
from src.tools.anomaly_tool import detect_anomalies, detect_structuring
from src.tools.eda_tool import run_eda
from src.tools.explainer_tool import explain_risk
from src.tools.feature_tool import add_aml_features, customer_summary
from src.tools.filter_tool import filter_transactions


TOOL_LABELS = {
    "filter_tool": "Scope & filter transactions",
    "eda_tool": "Exploratory data analysis",
    "feature_tool": "AML feature engineering",
    "anomaly_tool_structuring": "Rule-based structuring detection",
    "anomaly_tool_ml": "ML anomaly detection (Isolation Forest)",
    "aggregation_tool": "Customer aggregation & thresholds",
    "risk_classifier": "Risk scoring & classification",
    "explainer_tool": "Explain flags & recommend escalation",
}


def _apply_scope(transactions: pd.DataFrame, entities: dict) -> pd.DataFrame:
    scoped = transactions.copy()
    reference_date = scoped["timestamp"].max()

    if entities.get("days"):
        scoped = filter_transactions(
            scoped,
            date_start=reference_date - timedelta(days=int(entities["days"])),
        )

    return filter_transactions(
        scoped,
        customer_id=entities.get("customer_id"),
        country=entities.get("country"),
        min_amount=entities.get("min_amount"),
        max_amount=entities.get("max_amount"),
    )


def _detect_velocity_flags(transactions: pd.DataFrame) -> pd.DataFrame:
    features = add_aml_features(transactions)
    return features[
        (features["txn_count_24h"] >= 8)
        | ((features["txn_count_48h"] >= 12) & features["is_high_risk_country"])
    ].copy()


def execute_plan(transactions: pd.DataFrame, plan: ParsedQuery) -> dict[str, object]:
    """Run only the tools selected for this query and return analyst-ready output."""
    if plan.intent == "empty_query":
        return {
            "execution_plan": plan.to_plan(),
            "summary": "Enter a question to begin your AML investigation.",
            "alerts": [],
            "supporting_data": pd.DataFrame(),
            "eda": None,
            "metrics": {},
        }

    invoked = plan.tools_to_invoke
    entities = plan.entities
    working_df = _apply_scope(transactions, entities) if "filter_tool" in invoked else transactions.copy()

    results: dict[str, object] = {
        "execution_plan": plan.to_plan(),
        "tool_labels": {tool: TOOL_LABELS.get(tool, tool) for tool in invoked},
        "eda": None,
        "alerts": [],
        "supporting_data": pd.DataFrame(),
        "metrics": {},
        "summary": "",
        "analyst_guidance": "",
    }

    if working_df.empty:
        results["summary"] = "No transactions matched the filters extracted from your query."
        return results

    if "eda_tool" in invoked:
        results["eda"] = run_eda(working_df)

    if "feature_tool" in invoked:
        working_df = add_aml_features(working_df)

    flagged = pd.DataFrame()

    if plan.intent == "aggregation_query":
        min_count = int(entities.get("min_transaction_count") or 1)
        aggregated = aggregate_customers(
            working_df,
            max_amount=entities.get("max_amount"),
            min_amount=entities.get("min_amount"),
            min_transaction_count=min_count,
        )
        results["supporting_data"] = aggregated
        results["metrics"] = {
            "matching_customers": len(aggregated),
            "matching_transactions": int(aggregated["transaction_count"].sum()) if not aggregated.empty else 0,
        }

        if not aggregated.empty:
            top_customers = aggregated.head(10)["customer_id"].tolist()
            flagged = working_df[working_df["customer_id"].isin(top_customers)].copy()
            results["alerts"] = explain_risk(
                flagged,
                intent=plan.intent,
                original_query=plan.raw_query,
            )
            high_risk = [a for a in results["alerts"] if a["risk_level"] == "High"]
            results["summary"] = (
                f"Found **{len(aggregated)}** customers meeting your criteria "
                f"({min_count}+ transactions"
                + (f" under ${entities['max_amount']:,.0f}" if entities.get("max_amount") else "")
                + "). "
                f"Top result: **{aggregated.iloc[0]['customer_id']}** with "
                f"{int(aggregated.iloc[0]['transaction_count'])} transactions."
            )
            if high_risk:
                results["analyst_guidance"] = (
                    f"{len(high_risk)} customer(s) warrant immediate review. "
                    "Prioritise cases with the highest transaction counts and cross-jurisdiction activity."
                )
            else:
                results["analyst_guidance"] = (
                    "Review the top customers by transaction count. "
                    "Consider EDD for repeat near-threshold activity even when individual alerts are medium risk."
                )
        else:
            results["summary"] = "No customers matched the aggregation criteria in your query."
        return results

    if "anomaly_tool_structuring" in invoked:
        flagged = detect_structuring(working_df)

    if "anomaly_tool_ml" in invoked:
        scored = detect_anomalies(working_df)
        ml_flagged = scored[scored["ml_flag"] == "ML_ANOMALY"]
        flagged = ml_flagged if flagged.empty else pd.concat([flagged, ml_flagged]).drop_duplicates(
            subset=["transaction_id"]
        )

    if plan.intent == "velocity_detection":
        flagged = _detect_velocity_flags(working_df)

    if plan.intent == "eda_exploration" and flagged.empty:
        summary = customer_summary(working_df)
        results["supporting_data"] = summary.sort_values("near_threshold_ratio", ascending=False).head(20)
        results["metrics"] = results["eda"] or {}
        results["summary"] = (
            f"Dataset profile complete: **{results['eda']['total_transactions']:,}** transactions across "
            f"**{results['eda']['unique_customers']:,}** customers. "
            f"Mean transaction value: **${results['eda']['amount_stats']['mean']:,.2f}**. "
            "Review customers with elevated near-threshold ratios in the table below."
        )
        results["analyst_guidance"] = (
            "Use the channel and amount distributions to establish baseline behaviour. "
            "Follow up with a targeted structuring or velocity query on high-ratio customers."
        )
        return results

    if plan.intent == "customer_investigation" and flagged.empty:
        features = working_df if "txn_count_24h" in working_df.columns else add_aml_features(working_df)
        velocity = features[(features["txn_count_24h"] >= 8) | features["is_high_risk_country"]]
        if not velocity.empty:
            flagged = velocity

    if not flagged.empty:
        results["alerts"] = explain_risk(
            flagged,
            intent=plan.intent,
            original_query=plan.raw_query,
        )
        results["supporting_data"] = flagged.sort_values("timestamp", ascending=False)
        high = sum(1 for alert in results["alerts"] if alert["risk_level"] == "High")
        medium = sum(1 for alert in results["alerts"] if alert["risk_level"] == "Medium")
        results["metrics"] = {
            "flagged_transactions": len(flagged),
            "flagged_customers": len(results["alerts"]),
            "high_risk": high,
            "medium_risk": medium,
        }
        top = results["alerts"][0]
        results["summary"] = (
            f"Analysis complete: **{len(results['alerts'])}** customer(s) flagged across "
            f"**{len(flagged)}** transactions. "
            f"Highest priority: **{top['customer_id']}** ({top['risk_level']} risk)."
        )
        results["analyst_guidance"] = top["analyst_advice"]
    else:
        customer = entities.get("customer_id")
        if customer:
            results["summary"] = (
                f"No structuring, velocity, or ML alerts found for **{customer}** "
                "in the scoped transaction history. Continue periodic monitoring."
            )
            results["analyst_guidance"] = (
                "No immediate escalation required. Add to standard monitoring and "
                "re-run if new near-threshold or cross-border activity appears."
            )
            results["supporting_data"] = working_df.sort_values("timestamp", ascending=False)
        elif plan.intent == "structuring_detection":
            results["summary"] = "No structuring patterns detected in the requested scope."
            results["analyst_guidance"] = (
                "No SAR trigger at this time. Consider widening the date window or "
                "running an ML anomaly scan for non-structuring outliers."
            )
        else:
            results["summary"] = "Analysis complete — no suspicious patterns matched the selected detection path."
            results["analyst_guidance"] = (
                "Try narrowing to a customer ID, extending the date range, or requesting an EDA baseline summary."
            )

    return results


def answer_query(transactions: pd.DataFrame, query: str) -> dict[str, object]:
    """Main entry point: parse intent, build plan, execute tools, return rich response."""
    plan = parse_query(query)
    return execute_plan(transactions, plan)
