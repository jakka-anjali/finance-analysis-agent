import streamlit as st
from src.agent.planner import generate_plan
from src.agent.executor import execute_agent_plan

st.title("🛡️ AI-Powered AML Suspicious Activity Agent")

query = st.text_input("Ask the Compliance Agent:", "Is customer CUST_9999_STRUCTURER suspicious?")

if st.button("Run Agent"):
    with st.spinner("Agent planning execution path..."):
        # 1. Generate plan
        plan = generate_plan(query)
        
        # 2. Display Dynamic Agent Plan
        with st.expander("🤖 Agent Execution Plan & Tool Trace", expanded=True):
            st.json(plan)
            
        # 3. Execute plan
        output = execute_agent_plan(plan)
        
        # 4. Display Results
        st.subheader("Compliance Alerts & SAR Recommendations")
        alerts = output["results"].get("alerts", [])
        if alerts:
            st.dataframe(pd.DataFrame(alerts))
        else:
            st.success("No suspicious activity detected for this query scope.")