"""Agent Executor that dynamically runs selected tools based on planner instructions."""

from __future__ import annotations

import pandas as pd

from src.agent.intent_parser import ParsedQuery, parse_query
from src.agent.orchestrator import execute_plan


def execute_agent_plan(plan: dict, data_path=None, transactions: pd.DataFrame | None = None) -> dict:
    """Execute a planner-produced or pre-built JSON plan against transaction data."""
    if transactions is None:
        from src.tools.filter_tool import load_transactions

        transactions = load_transactions(data_path) if data_path else load_transactions()

    if isinstance(plan, ParsedQuery):
        parsed = plan
    elif "intent" in plan and "tools_to_invoke" in plan:
        parsed = ParsedQuery(
            raw_query=plan.get("original_query", ""),
            intent=plan.get("intent", "general_aml_analysis"),
            entities=plan.get("extracted_entities", {}),
            tools_to_invoke=plan.get("tools_to_invoke", []),
            tools_skipped=plan.get("skipped_tools", []),
            rationale=plan.get("rationale", ""),
        )
    else:
        parsed = parse_query(plan.get("original_query", str(plan)))

    result = execute_plan(transactions, parsed)
    return {
        "execution_trace": parsed.to_plan(),
        "results": result,
    }
