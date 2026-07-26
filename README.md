# FinTrace Copilot – Agentic AML Investigation System

An intelligent **agentic Anti-Money Laundering (AML)** investigation system that dynamically analyzes financial transactions based on natural-language analyst queries.

Instead of executing a fixed analytics pipeline, the agent first understands the analyst's intent, constructs a query-specific execution plan, invokes **only the required analytical tools**, and produces an explainable investigation report with risk assessment and escalation recommendations.

---

# Problem Statement

Financial institutions are required by regulatory authorities such as **FinCEN**, **FATF**, and national banking regulators to monitor financial transactions for suspicious activities.

Traditional AML systems rely heavily on static rule-based pipelines that:

- Generate excessive false positives
- Execute unnecessary analyses for every investigation
- Increase compliance costs
- Provide poor explainability
- Struggle to adapt to different analyst queries

The objective of this project is to build an **agentic AML investigation system** capable of understanding analyst intent and dynamically orchestrating only the analyses required to answer that specific investigation.

---

# Solution Overview

FinTrace Copilot behaves like an intelligent AML investigation assistant.

Given a natural-language query such as:

> "Find structuring patterns in the last 30 days"

or

> "Is customer CUST_4521 suspicious?"

the system automatically:

1. Understands the analyst's intent
2. Extracts relevant entities
3. Builds an execution plan
4. Invokes only the required AML tools
5. Skips unnecessary computation
6. Produces an explainable investigation report

Unlike conventional sequential workflows, the execution path changes dynamically depending on the analyst's request.

---

# Key Features

- Natural-language AML investigation
- Intent classification using lightweight parsing
- Automatic entity extraction
- Dynamic tool orchestration
- Conditional execution (only required tools run)
- Rule-based structuring detection
- Isolation Forest anomaly detection
- AML feature engineering
- Customer aggregation analysis
- Dataset profiling (EDA)
- Explainable risk scoring
- Human-readable investigation reports
- Execution trace showing invoked and skipped tools

---

# Agent Workflow

```
Analyst Query
        ↓
Intent Parser
        ↓
Entity Extraction
        ↓
Execution Plan
        ↓
Dynamic Orchestrator
        ↓
Conditional Tool Invocation
        ↓
Risk Classification
        ↓
Explainable Investigation Report
```

The execution path is **adaptive**, meaning different analyst queries invoke different combinations of tools.

---

# Project Architecture

The project is organized into modular components.

```
finance-analysis-agent/

│
├── app.py
├── requirements.txt
│
├── data/
│   └── aml_transactions.csv
│
├── prompts/
│   └── planner_prompt.txt
│
├── src/
│   │
│   ├── agent/
│   │      intent_parser.py
│   │      orchestrator.py
│   │      executor.py
│   │      planner.py
│   │
│   ├── tools/
│   │      filter_tool.py
│   │      eda_tool.py
│   │      feature_tool.py
│   │      aggregation_tool.py
│   │      anomaly_tool.py
│   │      risk_tool.py
│   │      explainer_tool.py
│   │
│   └── utils/
│
└── test_agent.py
```

---

# Technology Stack

### Frontend

- Streamlit

### Backend

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Isolation Forest

### Visualization

- Plotly
- Matplotlib

### Agent Design

- Modular Tool Registry
- Dynamic Orchestrator
- Intent Parser
- Execution Planner

---

# Core Components

## 1. Intent Parser

Responsible for understanding analyst queries.

Functions include:

- Intent classification
- Entity extraction
- Customer identification
- Time-window extraction
- Amount threshold extraction
- Country extraction
- Pattern detection

Produces a structured `ParsedQuery` object containing:

- intent
- extracted entities
- tools_to_invoke
- skipped_tools
- rationale

---

## 2. Dynamic Orchestrator

Acts as the brain of the agent.

Responsibilities include:

- Reading execution plans
- Routing execution dynamically
- Invoking only selected tools
- Skipping unnecessary analysis
- Maintaining execution trace
- Aggregating intermediate outputs

Unlike traditional pipelines, the orchestrator does **not** execute every module.

---

## 3. Tool Registry

Available analytical tools include:

### Filter Tool

Scopes the transaction dataset using:

- customer ID
- country
- amount thresholds
- time window

---

### EDA Tool

Performs exploratory analysis including:

- dataset profiling
- distributions
- descriptive statistics

Executed only when requested.

---

### Feature Engineering Tool

Generates AML-specific features including:

- rolling transaction counts
- near-threshold indicators
- transaction velocity metrics

---

### Rule-Based Structuring Detection

Detects suspicious structuring behaviour using regulatory-inspired heuristics.

---

### Isolation Forest Anomaly Detection

Uses unsupervised machine learning to identify anomalous transactions.

---

### Aggregation Tool

Produces customer-level summaries including:

- transaction counts
- aggregated transaction values
- threshold analysis

---

### Risk Classification Engine

Assigns:

- Risk Score
- Risk Level

Categories:

- Low
- Medium
- High

---

### Explainer Tool

Generates analyst-friendly explanations describing:

- why an entity was flagged
- suspicious patterns detected
- supporting evidence
- recommended escalation

---

# Dynamic Execution

One of the key innovations of this project is **dynamic execution**.

Example:

### Query

```
Find structuring patterns in the last 30 days
```

Executed:

- Filter
- Feature Engineering
- Structuring Detection
- Risk Classification
- Explanation

Skipped:

- Aggregation
- EDA
- Isolation Forest

---

### Query

```
Which customers made more than 10 transactions under $10,000?
```

Executed:

- Filter
- Aggregation
- Risk Classification
- Explanation

Skipped:

- Feature Engineering
- Structuring Detection
- Isolation Forest
- EDA

---

### Query

```
Explore this dataset
```

Executed:

- Filter
- EDA

Skipped:

- Structuring Detection
- Aggregation
- Isolation Forest

---

# Example Output

The generated investigation report includes:

- Execution Summary
- Execution Trace
- Tools Invoked
- Skipped Tools
- Flagged Customers
- Flagged Transactions
- Risk Score
- Risk Level
- Explanation
- Recommended Escalation
- Supporting Tables
- EDA Metrics (when applicable)

---

# Dataset

The project uses a sample AML transaction dataset located at:

```
data/aml_transactions.csv
```

The dataset contains transaction-level information including:

- Transaction ID
- Customer ID
- Timestamp
- Amount
- Country
- Transaction Channel

---

# Installation

Clone the repository

```bash
git clone https://github.com/jakka-anjali/finance-analysis-agent.git
cd finance-analysis-agent
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# Future Enhancements

- Real-time transaction monitoring
- Graph-based money laundering detection
- Multi-agent investigation workflows
- LLM-powered reasoning and explanations
- Regulatory report generation (SAR/STR)
- Streaming data support
- Case management integration
- Human feedback learning

---

# Contributors

**Jakka Anjali**

VIT Vellore

B.Tech Computer Science (IoT)

---

# License

This project was developed for educational and hackathon purposes.
