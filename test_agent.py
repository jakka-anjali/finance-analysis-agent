"""Comprehensive verification script for the AML Agent Orchestrator."""
from src.agent.planner import plan_execution
from src.agent.executor import execute_agent_plan
def test_pipeline():
    # [1/3] Targeted Entity Lookup
    query_1 = "Is customer CUST_9999_STRUCTURER suspicious?"
    print(f"\n[1/3] Testing Query: '{query_1}'")
    plan_1 = plan_execution(query_1)
    output_1 = execute_agent_plan(plan_1)
    alerts_1 = output_1.get("results", {}).get("alerts", [])
    print(f" -> Alerts Found: {len(alerts_1)}")
    if alerts_1:
        print(f"    Sample: {alerts_1[0]['explanation']}")

    # [2/3] Macro Pattern Analysis (Avoids broken relative date windows)
    query_2 = "Find all structuring patterns and smurfing rings in the dataset"
    print(f"\n[2/3] Testing Query: '{query_2}'")
    plan_2 = plan_execution(query_2)
    output_2 = execute_agent_plan(plan_2)
    alerts_2 = output_2.get("results", {}).get("alerts", [])
    print(f" -> Pattern Alerts Found: {len(alerts_2)}")
    if alerts_2:
        print(f"    Sample: {alerts_2[0]['explanation']}")

    # [3/3] Exploratory Data Analysis & High-Risk Profiling
    query_3 = "Run exploratory data analysis and show risk metrics"
    print(f"\n[3/3] Testing Query: '{query_3}'")
    plan_3 = plan_execution(query_3)
    output_3 = execute_agent_plan(plan_3)
    eda_res = output_3.get("results", {}).get("eda", {})
    print(f" -> EDA Metrics Generated: {list(eda_res.keys()) if eda_res else 'None'}")

if __name__ == "__main__":
    test_pipeline()
