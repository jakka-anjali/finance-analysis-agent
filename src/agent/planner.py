"""Planner module using LLM to extract intent and construct execution graphs."""

import json
import os
from openai import OpenAI

SYSTEM_PLANNER_PROMPT = """
You are the Lead AML Orchestrator Agent. Your task is to analyze user queries and produce a JSON execution plan choosing ONLY necessary tools.

Available Tools:
- "eda_tool": Run broad dataset statistics. (Use ONLY when user asks for overall data profiling/EDA).
- "filter_tool": Apply filters like customer_id, date range, amount, country.
- "feature_tool": Compute rolling transaction velocity and structuring indicators.
- "anomaly_tool_structuring": Detect threshold structuring ($9k-$9,999 cash deposits).
- "anomaly_tool_ml": Run Isolation Forest for general ML outliers.
- "explainer_tool": Generate compliance risk scores and escalation recommendations.

Rules:
1. For specific entity lookups (e.g., "Is CUST_1234 suspicious?"), DO NOT run eda_tool or anomaly_tool_ml.
2. For explicit rule queries (e.g., "txns under $10k"), run filter_tool + explainer_tool directly.
3. For general pattern queries (e.g., "Find structuring"), run filter_tool + feature_tool + anomaly_tool_structuring + explainer_tool.

Return ONLY a JSON object:
{
    "intent": "<INTENT_NAME>",
    "extracted_entities": {"customer_id": null, "date_range": null, "amount_threshold": null},
    "tools_to_invoke": ["tool1", "tool2"],
    "skipped_tools": ["tool3", "tool4"],
    "plan_reasoning": "<Short explanation of tool choice>"
}
"""

def generate_plan(user_query: str) -> dict:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PLANNER_PROMPT},
            {"role": "user", "content": user_query}
        ],
        temperature=0.0
    )
    return json.loads(response.choices[0].message.content)