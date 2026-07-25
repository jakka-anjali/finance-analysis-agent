"""Natural-language intent and entity extraction for AML analyst queries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


CUSTOMER_PATTERN = re.compile(r"\bCUST_\d{4}(?:_[A-Z]+)?\b", re.IGNORECASE)
DAYS_PATTERN = re.compile(r"(?:last|past|within)\s+(\d+)\s+days?", re.IGNORECASE)
AMOUNT_UNDER_PATTERN = re.compile(
    r"(?:under|below|less than|at or below|<=?)\s+\$?\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
AMOUNT_OVER_PATTERN = re.compile(
    r"(?:over|above|more than|greater than|>=?)\s+\$?\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
TXN_COUNT_PATTERN = re.compile(
    r"(\d+)\s*\+?\s*(?:or more\s+)?transactions?",
    re.IGNORECASE,
)
COUNTRY_PATTERN = re.compile(
    r"\b(?:in|from|country)\s+([A-Z]{2})\b|\bcountry\s+([A-Z]{2})\b",
    re.IGNORECASE,
)


@dataclass
class ParsedQuery:
    """Structured representation of an analyst's natural-language request."""

    raw_query: str
    intent: str
    entities: dict[str, Any] = field(default_factory=dict)
    tools_to_invoke: list[str] = field(default_factory=list)
    tools_skipped: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_plan(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "extracted_entities": self.entities,
            "tools_to_invoke": self.tools_to_invoke,
            "skipped_tools": self.tools_skipped,
            "rationale": self.rationale,
            "original_query": self.raw_query,
        }


ALL_TOOLS = [
    "filter_tool",
    "eda_tool",
    "feature_tool",
    "anomaly_tool_structuring",
    "anomaly_tool_ml",
    "aggregation_tool",
    "risk_classifier",
    "explainer_tool",
]


def _parse_amount(value: str) -> float:
    return float(value.replace(",", ""))


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _mentions_eda(text: str) -> bool:
    keywords = (
        "explor",
        "profile",
        "baseline",
        " overview",
        "summarize",
        "summarise",
        "understand the data",
        "understand this data",
        "dataset",
        " eda",
        "data quality",
        "distribution",
    )
    return any(keyword in text for keyword in keywords)


def _mentions_structuring(text: str) -> bool:
    keywords = (
        "structur",
        "smurf",
        "near threshold",
        "near-threshold",
        "9000",
        "9,000",
        "9999",
        "9,999",
        "10000",
        "10,000",
        "cash deposit",
        "threshold evasion",
    )
    return any(keyword in text for keyword in keywords)


def _mentions_ml(text: str) -> bool:
    keywords = (
        "anomal",
        "outlier",
        "machine learning",
        " ml ",
        "isolation forest",
        "unsupervised",
        "statistical",
    )
    return any(keyword in text for keyword in keywords)


def _mentions_velocity(text: str) -> bool:
    keywords = ("velocity", "rapid", "high frequency", "high-frequency", "quick cash")
    return any(keyword in text for keyword in keywords)


def _mentions_aggregation(text: str) -> bool:
    keywords = (
        "which customers",
        "how many customers",
        "customers made",
        "customers with",
        "count of",
        "top customers",
        "most transactions",
    )
    return bool(TXN_COUNT_PATTERN.search(text) or any(keyword in text for keyword in keywords))


def _mentions_customer_check(text: str) -> bool:
    if CUSTOMER_PATTERN.search(text):
        return True
    keywords = (
        "is customer",
        "customer id",
        "this customer",
        "suspicious",
        "risk profile",
        "risk assessment",
        "investigate",
    )
    return any(keyword in text for keyword in keywords)


def _extract_entities(query: str) -> dict[str, Any]:
    normalized = _normalize(query)
    entities: dict[str, Any] = {
        "customer_id": None,
        "days": None,
        "amount_threshold": None,
        "min_amount": None,
        "max_amount": None,
        "min_transaction_count": None,
        "country": None,
        "pattern_type": None,
    }

    customer_match = CUSTOMER_PATTERN.search(query)
    if customer_match:
        entities["customer_id"] = customer_match.group(0).upper()

    days_match = DAYS_PATTERN.search(normalized)
    if days_match:
        entities["days"] = int(days_match.group(1))

    under_match = AMOUNT_UNDER_PATTERN.search(normalized)
    if under_match:
        entities["max_amount"] = _parse_amount(under_match.group(1))
        entities["amount_threshold"] = entities["max_amount"]

    over_match = AMOUNT_OVER_PATTERN.search(normalized)
    if over_match:
        entities["min_amount"] = _parse_amount(over_match.group(1))

    txn_match = TXN_COUNT_PATTERN.search(normalized)
    if txn_match:
        entities["min_transaction_count"] = int(txn_match.group(1))

    country_match = COUNTRY_PATTERN.search(query)
    if country_match:
        entities["country"] = (country_match.group(1) or country_match.group(2)).upper()

    if _mentions_structuring(normalized):
        entities["pattern_type"] = "structuring"
    elif _mentions_velocity(normalized):
        entities["pattern_type"] = "velocity"
    elif _mentions_ml(normalized):
        entities["pattern_type"] = "ml_anomaly"

    return entities


def _build_tool_plan(intent: str, entities: dict[str, Any]) -> tuple[list[str], list[str], str]:
    invoked: list[str] = []
    skipped: list[str] = []

    if intent == "customer_investigation":
        invoked = ["filter_tool", "feature_tool", "anomaly_tool_structuring", "risk_classifier", "explainer_tool"]
        skipped = ["eda_tool", "anomaly_tool_ml", "aggregation_tool"]
        rationale = (
            "Single-customer lookup: scoped filtering and targeted structuring checks "
            "without running full dataset EDA or ML scoring."
        )

    elif intent == "structuring_detection":
        invoked = ["filter_tool", "feature_tool", "anomaly_tool_structuring", "risk_classifier", "explainer_tool"]
        skipped = ["eda_tool", "anomaly_tool_ml"]
        if entities.get("days"):
            rationale = (
                f"Time-bounded structuring search ({entities['days']} days): "
                "feature engineering and rule-based detection only."
            )
        else:
            rationale = "Pattern-focused structuring search without broad EDA or ML anomaly scoring."

    elif intent == "aggregation_query":
        invoked = ["filter_tool", "aggregation_tool", "risk_classifier", "explainer_tool"]
        skipped = ["eda_tool", "feature_tool", "anomaly_tool_ml", "anomaly_tool_structuring"]
        rationale = (
            "Threshold aggregation query: direct customer roll-up and rule evaluation "
            "without ML anomaly detection."
        )

    elif intent == "eda_exploration":
        invoked = ["filter_tool", "eda_tool", "feature_tool", "risk_classifier", "explainer_tool"]
        skipped = ["anomaly_tool_ml"]
        rationale = "Broad exploration request: baseline profiling and feature summary for analyst context."

    elif intent == "ml_anomaly_scan":
        invoked = ["filter_tool", "feature_tool", "anomaly_tool_ml", "risk_classifier", "explainer_tool"]
        skipped = ["eda_tool", "anomaly_tool_structuring"]
        rationale = "ML-focused anomaly scan: statistical outlier detection on filtered transactions."

    elif intent == "velocity_detection":
        invoked = ["filter_tool", "feature_tool", "risk_classifier", "explainer_tool"]
        skipped = ["eda_tool", "anomaly_tool_ml"]
        rationale = "Velocity-focused analysis using rolling transaction frequency features."

    else:
        invoked = ["filter_tool", "eda_tool", "feature_tool", "anomaly_tool_structuring", "risk_classifier", "explainer_tool"]
        skipped = ["anomaly_tool_ml", "aggregation_tool"]
        rationale = "General AML analysis: balanced profiling, structuring detection, and explainable flags."

    return invoked, skipped, rationale


def _resolve_intent(normalized: str, entities: dict[str, Any]) -> str:
    if entities.get("customer_id") or (
        _mentions_customer_check(normalized) and entities.get("customer_id")
    ):
        return "customer_investigation"

    if entities.get("customer_id"):
        return "customer_investigation"

    if _mentions_aggregation(normalized):
        return "aggregation_query"

    if _mentions_eda(normalized) and not _mentions_structuring(normalized) and "suspicious" not in normalized:
        return "eda_exploration"

    if _mentions_structuring(normalized):
        return "structuring_detection"

    if _mentions_ml(normalized):
        return "ml_anomaly_scan"

    if _mentions_velocity(normalized):
        return "velocity_detection"

    if entities.get("customer_id"):
        return "customer_investigation"

    if "suspicious" in normalized and entities.get("customer_id"):
        return "customer_investigation"

    return "general_aml_analysis"


def parse_query(query: str) -> ParsedQuery:
    """Convert a natural-language AML query into a dynamic execution plan."""
    normalized = _normalize(query)
    if not normalized:
        return ParsedQuery(
            raw_query=query,
            intent="empty_query",
            rationale="No query provided.",
        )

    entities = _extract_entities(query)

    if CUSTOMER_PATTERN.search(query):
        entities["customer_id"] = CUSTOMER_PATTERN.search(query).group(0).upper()
        intent = "customer_investigation"
    elif _mentions_aggregation(normalized):
        intent = "aggregation_query"
    elif (
        _mentions_eda(normalized)
        and not (_mentions_structuring(normalized) or entities.get("days"))
        and "suspicious" not in normalized
        and "anomal" not in normalized
    ):
        intent = "eda_exploration"
    elif _mentions_structuring(normalized):
        intent = "structuring_detection"
    elif _mentions_ml(normalized):
        intent = "ml_anomaly_scan"
    elif _mentions_velocity(normalized):
        intent = "velocity_detection"
    elif "suspicious" in normalized and entities.get("customer_id"):
        intent = "customer_investigation"
    elif re.search(r"\bis\s+customer\b", normalized) or re.search(r"\bcustomer\s+id\b", normalized):
        intent = "customer_investigation"
    else:
        intent = _resolve_intent(normalized, entities)
        if "suspicious" in normalized or "money laundering" in normalized:
            intent = "general_aml_analysis"

    invoked, skipped, rationale = _build_tool_plan(intent, entities)

    return ParsedQuery(
        raw_query=query,
        intent=intent,
        entities=entities,
        tools_to_invoke=invoked,
        tools_skipped=skipped,
        rationale=rationale,
    )
