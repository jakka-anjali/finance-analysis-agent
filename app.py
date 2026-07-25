"""Streamlit dashboard for investigating synthetic AML transactions."""

from datetime import timedelta

import pandas as pd
import streamlit as st

from src.tools.anomaly_tool import detect_anomalies, detect_structuring
from src.tools.eda_tool import run_eda
from src.tools.explainer_tool import explain_risk
from src.tools.feature_tool import add_aml_features
from src.tools.filter_tool import filter_transactions, load_transactions


st.set_page_config(page_title="AML Sentinel", page_icon="🛡️", layout="wide")


@st.cache_data
def get_transactions() -> pd.DataFrame:
    """Load the dataset once per dashboard session."""
    return load_transactions()


@st.cache_data
def get_ml_anomalies(transactions: pd.DataFrame) -> pd.DataFrame:
    """Run the Isolation Forest once, rather than on every interaction."""
    return detect_anomalies(transactions)


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


try:
    transactions = get_transactions()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Could not load the AML dataset: {error}")
    st.stop()


st.title("🛡️ AML Sentinel")
st.caption("Synthetic transaction-monitoring dashboard for hackathon demonstrations.")

with st.sidebar:
    st.header("Transaction filters")
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

overview_tab, investigation_tab, alerts_tab = st.tabs(
    ["Overview", "Customer investigation", "Alerts"]
)

with overview_tab:
    profile = run_eda(filtered_transactions)
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Transactions", f"{profile['total_transactions']:,}")
    metric_2.metric("Customers", f"{profile['unique_customers']:,}")
    metric_3.metric("Total value", format_currency(filtered_transactions["amount"].sum()))
    metric_4.metric("High-risk jurisdiction transactions", profile["high_risk_jurisdictions_count"])

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Transactions by channel")
        st.bar_chart(pd.Series(profile["channels_distribution"], name="transactions"))
    with chart_right:
        st.subheader("Labels in synthetic data")
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

        if st.button("Run risk assessment", type="primary"):
            structuring_alerts = detect_structuring(customer_transactions)
            explanations = explain_risk(structuring_alerts)
            if explanations:
                for alert in explanations:
                    st.error(f"{alert['risk_level']} risk — {alert['recommended_action']}")
                    st.write(alert["explanation"])
            else:
                st.success("No rule-based structuring pattern was found for this customer.")

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
        for alert in explain_risk(structuring_alerts):
            st.write(f"**{alert['customer_id']}** — {alert['explanation']}")

    st.subheader("Highest-scoring ML anomalies")
    ml_anomalies = get_ml_anomalies(transactions)
    visible_anomalies = ml_anomalies[ml_anomalies["ml_flag"] == "ML_ANOMALY"].head(20)
    st.dataframe(
        visible_anomalies[
            ["transaction_id", "customer_id", "timestamp", "amount", "channel", "country", "anomaly_score"]
        ],
        use_container_width=True,
        hide_index=True,
    )
