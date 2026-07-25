# finance-analysis-agent

AI-powered suspicious-activity detection agent that dynamically orchestrates EDA, feature engineering, anomaly detection, risk classification, and human-readable explanations for transaction and customer-level AML analysis.

## Summary
This project demonstrates an agentic system that accepts natural-language queries (e.g., "Find structuring in the last 30 days" or "Is customer 4521 suspicious?"), builds a query-aware execution plan, invokes only the necessary analytic tools, and returns ranked suspicious items with risk levels, explanations, and recommended escalation actions (monitor / review / report).

Key capabilities:
- Intent and filter extraction from free-text queries (date ranges, customer IDs, transaction types).
- Dynamic planner that invokes only required tools (no fixed sequential pipeline).
- On-demand EDA, targeted preprocessing, and AML feature engineering (frequency, rolling sums, velocity).
- Hybrid detection (rules + statistical / ML scoring).
- Risk classification into low / medium / high and concise human-readable explanations.

## Repo layout
```
app.py                    # Streamlit demo / quick UI
requirements.txt          # Python dependencies
data/
  aml_transactions.csv    # Sample dataset (included)
prompts/
  planner_prompt.txt      # Planner/agent prompt templates
src/
  main.py                 # CLI / example entrypoint
  agent/                  # Agent core: parser, planner, orchestrator, executor
  tools/                  # Modular tools: eda, filter, features, anomaly, risk, explainer
  utils/                  # Helper utilities
test_agent.py             # Basic tests / usage examples
```

## Quick start
1. Clone
```bash
git clone https://github.com/jakka-anjali/finance-analysis-agent.git
cd finance-analysis-agent
```

2. Virtual environment & install
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Run demo (Streamlit)
```bash
streamlit run app.py
```

4. Run example / tests
```bash
python src/main.py --help
pytest -q
```

Environment variables:
- If using the LLM-backed explainer, set OPENAI_API_KEY (or other provider key) in your environment.

## How it works (runtime overview)
1. User sends a natural-language query to the agent.
2. Intent parser extracts intent, filters, entities, and pattern types.
3. Planner creates a minimal execution plan (which tools, order, and data subset).
4. Orchestrator/Executor runs the selected tools and aggregates outputs.
5. Risk tool converts scores into categories; explainer produces human-readable reasons.
6. Results returned: execution summary, flagged items, risk level, explanation, suggested action, and optional supporting charts.

Example behaviors:
- "Find structuring patterns in the last 30 days" → apply time filter → run structuring features + detection → skip full EDA.
- "Which customers made 10+ transactions under $10,000?" → run aggregation/threshold rule only.
- "Is customer 4521 suspicious?" → perform single-entity feature compute, scoring, and explain.

## Data & privacy
- This repository currently contains `data/aml_transactions.csv`. The repository is public by default: anyone who can access the repo URL can view and clone all files, including that CSV.
- To prevent public access, change the repository visibility to private or remove the dataset from the repository and its history. Note: removing a file from the current commit does not erase it from git history — use `git filter-repo` or BFG to purge history if necessary.

Quick removal (non-history rewriting):
```bash
git rm --cached data/aml_transactions.csv
echo "data/aml_transactions.csv" >> .gitignore
git commit -m "remove dataset from current branch and ignore it"
git push
```
To permanently remove from history, follow GitHub's documentation or use BFG/git-filter-repo and force-push (coordinate with collaborators).

## Configuration & customization
- Detection rules, thresholds, and classification logic live in `src/tools/anomaly_tool.py` and `src/tools/risk_tool.py`.
- Add/replace models by extending `src/tools/` and updating the planner to reference new tools.
- Prompts for planner/LLM components are in `prompts/`.

## Contributing
- Open an issue describing the feature/bug before large changes.
- Add tests for new behaviors and follow the existing module structure.
- Add a LICENSE file to declare reuse terms.

## License
- No license file included.

## Contact
- Repo owner: jakka-anjali

