"""FinTrace Copilot — AI-powered AML analyst assistant."""

from __future__ import annotations

from datetime import timedelta
from io import BytesIO

import pandas as pd
import streamlit as st

from src.agent.orchestrator import TOOL_LABELS, answer_query
from src.tools.anomaly_tool import detect_anomalies, detect_structuring
from src.tools.eda_tool import run_eda
from src.tools.explainer_tool import explain_risk
from src.tools.filter_tool import filter_transactions, load_transactions

APP_NAME = "FinTrace Copilot"
APP_TAGLINE = (
    "Your AI investigation partner for suspicious activity detection, "
    "explainable risk scoring, and analyst-ready escalation guidance."
)

EXAMPLE_QUERIES = [
    "Analyse this dataset for suspicious activity",
    "Find structuring patterns in the last 60 days",
    "Which customers made 10+ transactions under $10,000?",
    "Is CUST_9999_STRUCTURER suspicious?",
    "Run ML anomaly detection on high-risk jurisdictions",
    "Show me rapid velocity patterns in the last 30 days",
]

CAPABILITY_CHECKLIST = [
    ("Natural-language intent parsing", True, "Extracts intent, filters, entities, and AML pattern types from queries"),
    ("Dynamic execution planning", True, "Builds a query-specific tool plan — skips unnecessary steps"),
    ("Selective EDA", True, "Runs profiling only when exploration is requested, not for targeted lookups"),
    ("AML feature engineering", True, "Near-threshold flags, rolling velocity, jurisdiction risk"),
    ("Rule-based structuring detection", True, "3+ cash deposits $9k–$9,999 within 48 hours"),
    ("ML anomaly detection", True, "Isolation Forest on amount, timing, channel, and jurisdiction"),
    ("Customer aggregation queries", True, "Threshold rules like '10+ transactions under $10k'"),
    ("Risk classification (Low/Med/High)", True, "Context-aware scoring with escalation mapping"),
    ("Explainable flags tied to query", True, "Each alert explains why it was flagged for this investigation"),
    ("Analyst escalation guidance", True, "Monitor / Review / Report with playbook next steps"),
    ("External CSV/Excel upload", True, "Upload custom datasets; demo defaults to bundled AML CSV"),
    ("LLM planner (optional)", False, "OpenRouter planner available via test_agent.py when API key is set"),
    ("Live SAR filing integration", False, "Escalation recommendations only — no external compliance system"),
    ("Real-time streaming transactions", False, "Batch analysis on uploaded or bundled datasets"),
]

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0c4a6e 100%);
        border-radius: 16px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.5rem;
        color: #f8fafc;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .hero h1 { color: #f8fafc !important; font-size: 2rem !important; margin-bottom: 0.3rem !important; font-weight: 700 !important; }
    .hero p { color: #cbd5e1 !important; font-size: 1.02rem !important; margin: 0 !important; line-height: 1.55 !important; }

    .risk-high { background: #fef2f2; border-left: 4px solid #dc2626; padding: 0.85rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; }
    .risk-medium { background: #fffbeb; border-left: 4px solid #d97706; padding: 0.85rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; }
    .risk-low { background: #f0fdf4; border-left: 4px solid #16a34a; padding: 0.85rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; }

    .plan-chip { display: inline-block; background: #e0f2fe; color: #0369a1; padding: 0.25rem 0.65rem; border-radius: 999px; font-size: 0.82rem; margin: 0.15rem 0.25rem 0.15rem 0; font-weight: 500; }
    .skip-chip { display: inline-block; background: #f1f5f9; color: #64748b; padding: 0.25rem 0.65rem; border-radius: 999px; font-size: 0.82rem; margin: 0.15rem 0.25rem 0.15rem 0; text-decoration: line-through; }

    .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.9rem 1rem; text-align: center; }
    .metric-card .val { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
    .metric-card .lbl { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }

    div[data-testid="stSidebar"] { background: #f8fafc; }
</style>
"""


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def risk_badge(level: str) -> str:
    colours = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    return f"{colours.get(level, '⚪')} {level}"


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    buffer = BytesIO(uploaded_file.getvalue())
    if name.endswith(".csv"):
        return load_transactions_from_buffer(buffer, "csv")
    if name.endswith((".xlsx", ".xls")):
        return load_transactions_from_buffer(buffer, "excel")
    raise ValueError("Supported formats: .csv, .xlsx, .xls")


def load_transactions_from_buffer(buffer: BytesIO, kind: str) -> pd.DataFrame:
    if kind == "csv":
        df = pd.read_csv(buffer)
    else:
        df = pd.read_excel(buffer)

    required = {"transaction_id", "customer_id", "timestamp", "amount", "channel", "country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["amount"] = pd.to_numeric(df["amount"])
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data
def get_default_transactions() -> pd.DataFrame:
    return load_transactions()


@st.cache_data
def get_ml_anomalies(_transactions: pd.DataFrame) -> pd.DataFrame:
    return detect_anomalies(_transactions)


def render_execution_plan(plan: dict) -> None:
    st.markdown("#### Agent execution plan")
    st.caption(plan.get("rationale", ""))

    invoked = plan.get("tools_to_invoke", [])
    skipped = plan.get("skipped_tools", [])

    chip_html = "".join(
        f'<span class="plan-chip">{TOOL_LABELS.get(t, t)}</span>' for t in invoked
    )
    skip_html = "".join(
        f'<span class="skip-chip">{TOOL_LABELS.get(t, t)}</span>' for t in skipped
    )

    st.markdown(f"**Invoked:** {chip_html}", unsafe_allow_html=True)
    if skipped:
        st.markdown(f"**Skipped:** {skip_html}", unsafe_allow_html=True)

    entities = plan.get("extracted_entities", {})
    active_entities = {k: v for k, v in entities.items() if v is not None}
    if active_entities:
        st.json(active_entities)


def render_agent_response(response: dict) -> None:
    st.markdown(f"### {response.get('summary', 'Analysis complete')}")

    if response.get("analyst_guidance"):
        st.info(response["analyst_guidance"])

    plan = response.get("execution_plan", {})
    with st.expander("View agent workflow & detected entities", expanded=False):
        render_execution_plan(plan)

    eda = response.get("eda")
    if eda and isinstance(eda, dict) and "total_transactions" in eda:
        st.markdown("#### Dataset baseline (EDA)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", f"{eda['total_transactions']:,}")
        c2.metric("Customers", f"{eda['unique_customers']:,}")
        c3.metric("Mean amount", format_currency(eda["amount_stats"]["mean"]))
        c4.metric("High-risk jurisdictions", eda["high_risk_jurisdictions_count"])

        chart_l, chart_r = st.columns(2)
        with chart_l:
            st.bar_chart(pd.Series(eda["channels_distribution"], name="By channel"))
        with chart_r:
            if eda.get("label_distribution"):
                st.bar_chart(pd.Series(eda["label_distribution"], name="By label"))

    metrics = response.get("metrics", {})
    if metrics and "flagged_customers" in metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Flagged customers", metrics["flagged_customers"])
        m2.metric("High risk", metrics.get("high_risk", 0))
        m3.metric("Medium risk", metrics.get("medium_risk", 0))

    alerts = response.get("alerts", [])
    if alerts:
        st.markdown("#### Flagged entities & analyst recommendations")
        for alert in alerts:
            level = alert["risk_level"]
            css_class = f"risk-{level.lower()}"
            st.markdown(
                f'<div class="{css_class}"><strong>{risk_badge(level)}</strong> — '
                f'<strong>{alert["customer_id"]}</strong> '
                f'(score {alert.get("risk_score", "—")}/100)<br>'
                f'{alert["explanation"]}<br><br>'
                f'<strong>Escalation:</strong> {alert["recommended_action"]}</div>',
                unsafe_allow_html=True,
            )
            with st.expander(f"Analyst playbook — {alert['customer_id']}"):
                for step in alert.get("playbook_steps", []):
                    st.markdown(f"- {step}")

    data = response.get("supporting_data")
    if isinstance(data, pd.DataFrame) and not data.empty:
        st.markdown("#### Supporting evidence")
        st.dataframe(data.head(100), use_container_width=True, hide_index=True)


st.set_page_config(page_title=APP_NAME, page_icon="🔍", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    f'<div class="hero"><h1>🔍 {APP_NAME}</h1><p>{APP_TAGLINE}</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data source")
    data_source = st.radio(
        "Dataset",
        ["Demo dataset (aml_transactions.csv)", "Upload CSV / Excel"],
        index=0,
    )

    uploaded_df = None
    if data_source == "Upload CSV / Excel":
        uploaded = st.file_uploader("Upload transactions file", type=["csv", "xlsx", "xls"])
        if uploaded:
            try:
                uploaded_df = load_uploaded_file(uploaded)
                st.success(f"Loaded {len(uploaded_df):,} transactions from {uploaded.name}")
            except (ValueError, FileNotFoundError) as err:
                st.error(str(err))
        else:
            st.caption("Required columns: transaction_id, customer_id, timestamp, amount, channel, country")
    else:
        st.caption("Using bundled synthetic AML dataset for demonstration.")

    st.divider()
    st.header("Transaction filters")

try:
    transactions = uploaded_df if uploaded_df is not None else get_default_transactions()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Could not load the AML dataset: {error}")
    st.stop()

with st.sidebar:
    date_range = st.date_input(
        "Date range",
        value=(transactions["timestamp"].min().date(), transactions["timestamp"].max().date()),
        min_value=transactions["timestamp"].min().date(),
        max_value=transactions["timestamp"].max().date(),
    )
    selected_country = st.selectbox("Country", ["All countries", *sorted(transactions["country"].unique())])
    min_amount = st.number_input("Minimum amount", min_value=0.0, value=0.0, step=100.0)

date_start, date_end = date_range
filtered_transactions = filter_transactions(
    transactions,
    date_start=date_start,
    date_end=pd.Timestamp(date_end) + timedelta(days=1) - timedelta(microseconds=1),
    country=None if selected_country == "All countries" else selected_country,
    min_amount=min_amount or None,
)

assistant_tab, overview_tab, investigation_tab, alerts_tab, capabilities_tab = st.tabs(
    ["Investigate", "Overview", "Customer deep-dive", "Alert queue", "Capabilities"]
)

with assistant_tab:
    st.subheader("Ask the AML analyst copilot")
    st.caption(
        "The agent parses your query, builds a dynamic execution plan, and invokes only the tools needed — "
        "no fixed pipeline."
    )

    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    query_cols = st.columns([5, 1])
    with query_cols[0]:
        query = st.text_input(
            "Your investigation question",
            value=st.session_state.last_query,
            placeholder="Find structuring patterns in the last 30 days",
            label_visibility="collapsed",
        )
    with query_cols[1]:
        analyse = st.button("Run analysis", type="primary", use_container_width=True)

    st.markdown("**Try an example:**")
    example_cols = st.columns(3)
    for idx, example in enumerate(EXAMPLE_QUERIES):
        if example_cols[idx % 3].button(example, key=f"ex_{idx}", use_container_width=True):
            st.session_state.last_query = example
            query = example
            analyse = True

    if analyse and query.strip():
        st.session_state.last_query = query
        with st.spinner("Agent is parsing intent and orchestrating tools..."):
            response = answer_query(filtered_transactions, query)
        render_agent_response(response)

with overview_tab:
    profile = run_eda(filtered_transactions)
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Transactions", f"{profile['total_transactions']:,}")
    metric_2.metric("Customers", f"{profile['unique_customers']:,}")
    metric_3.metric("Total value", format_currency(filtered_transactions["amount"].sum()))
    metric_4.metric("High-risk jurisdiction txns", profile["high_risk_jurisdictions_count"])

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Transactions by channel")
        st.bar_chart(pd.Series(profile["channels_distribution"], name="transactions"))
    with chart_right:
        st.subheader("Labels in dataset")
        if profile.get("label_distribution"):
            st.bar_chart(pd.Series(profile["label_distribution"], name="transactions"))

    st.subheader("Filtered transactions")
    st.dataframe(
        filtered_transactions.sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

with investigation_tab:
    customer_id = st.selectbox("Select a customer", sorted(transactions["customer_id"].unique()))
    customer_transactions = filter_transactions(filtered_transactions, customer_id=customer_id)

    if customer_transactions.empty:
        st.info("This customer has no transactions matching the selected filters.")
    else:
        customer_metrics = st.columns(3)
        customer_metrics[0].metric("Transactions", len(customer_transactions))
        customer_metrics[1].metric("Total value", format_currency(customer_transactions["amount"].sum()))
        customer_metrics[2].metric("Largest transaction", format_currency(customer_transactions["amount"].max()))
        st.dataframe(customer_transactions, use_container_width=True, hide_index=True)

        if st.button("Run full risk assessment", type="primary"):
            response = answer_query(
                customer_transactions,
                f"Is customer {customer_id} suspicious?",
            )
            render_agent_response(response)

with alerts_tab:
    st.subheader("Structuring alerts")
    structuring_alerts = detect_structuring(filtered_transactions)
    if structuring_alerts.empty:
        st.success("No structuring alerts match the selected filters.")
    else:
        st.warning(f"{len(structuring_alerts)} linked transactions triggered the structuring rule.")
        st.dataframe(
            structuring_alerts[["customer_id", "timestamp", "amount", "country", "txn_count_48h"]],
            use_container_width=True,
            hide_index=True,
        )
        for alert in explain_risk(structuring_alerts, intent="structuring_detection"):
            st.markdown(
                f"**{alert['customer_id']}** — {alert['risk_level']} risk — "
                f"{alert['recommended_action']}"
            )
            st.write(alert["explanation"])

    st.subheader("Top ML anomalies")
    ml_anomalies = get_ml_anomalies(transactions)
    visible_anomalies = ml_anomalies[ml_anomalies["ml_flag"] == "ML_ANOMALY"].head(20)
    st.dataframe(
        visible_anomalies[
            ["transaction_id", "customer_id", "timestamp", "amount", "channel", "country", "anomaly_score"]
        ],
        use_container_width=True,
        hide_index=True,
    )

with capabilities_tab:
    st.subheader("Agent capability checklist")
    st.caption("What this agentic AML system supports vs. what remains for production deployment.")

    done_count = sum(1 for _, done, _ in CAPABILITY_CHECKLIST if done)
    st.progress(done_count / len(CAPABILITY_CHECKLIST), text=f"{done_count}/{len(CAPABILITY_CHECKLIST)} capabilities implemented")

    for name, done, detail in CAPABILITY_CHECKLIST:
        icon = "✅" if done else "⬜"
        st.markdown(f"{icon} **{name}** — {detail}")

    st.divider()
    st.subheader("Agentic workflow")
    st.markdown(
        """
```mermaid
flowchart LR
    Q[Analyst query] --> P[Intent parser]
    P --> E[Entity extraction]
    E --> PL[Dynamic plan]
    PL --> T1[Filter]
    PL --> T2[EDA]
    PL --> T3[Features]
    PL --> T4[Detection]
    PL --> T5[Risk score]
    PL --> T6[Explain & advise]
    T6 --> R[Analyst-ready output]
```
        """
    )
    st.caption(
        "Each query takes a different path. Example: 'find structuring in last 30 days' skips full EDA; "
        "'which customers made 10+ txns under $10k' runs aggregation only, not ML."
    )
