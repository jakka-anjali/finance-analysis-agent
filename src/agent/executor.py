"""Agent Executor that dynamically runs selected tools based on planner instructions."""

import pandas as pd
from src.tools.filter_tool import load_transactions, filter_transactions
from src.tools.feature_tool import add_aml_features
from src.tools.anomaly_tool import detect_structuring, detect_anomalies
from src.tools.explainer_tool import explain_risk
from src.tools.eda_tool import run_eda

def execute_agent_plan(plan: dict, data_path=None) -> dict:
    df = load_transactions(data_path) if data_path else load_transactions()
    invoked_tools = plan.get("tools_to_invoke", [])
    entities = plan.get("extracted_entities", {})
    
    execution_results = {}
    working_df = df.copy()

    # Step 1: Filtering (if requested or entities extracted)
    if "filter_tool" in invoked_tools or any(entities.values()):
        working_df = filter_transactions(
            working_df,
            customer_id=entities.get("customer_id"),
            min_amount=entities.get("amount_threshold")
        )

    # Step 2: Selective Tool Execution
    if "eda_tool" in invoked_tools:
        execution_results["eda"] = run_eda(working_df)

    if "feature_tool" in invoked_tools:
        working_df = add_aml_features(working_df)

    if "anomaly_tool_structuring" in invoked_tools:
        flagged = detect_structuring(working_df)
        execution_results["flagged"] = flagged
        execution_results["alerts"] = explain_risk(flagged)

    elif "anomaly_tool_ml" in invoked_tools:
        scored = detect_anomalies(working_df)
        flagged = scored[scored["ml_flag"] == "ML_ANOMALY"]
        execution_results["flagged"] = flagged
        execution_results["alerts"] = explain_risk(flagged)

    elif "explainer_tool" in invoked_tools and "alerts" not in execution_results:
        execution_results["alerts"] = explain_risk(working_df)

    return {
        "execution_trace": plan,
        "results": execution_results
    }