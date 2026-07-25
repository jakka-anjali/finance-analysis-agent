"""Rule-based and machine-learning transaction anomaly detection."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .feature_tool import add_aml_features


def detect_structuring(transactions: pd.DataFrame, minimum_count: int = 3) -> pd.DataFrame:
    """Flag customers with at least `minimum_count` $9k-$9,999 deposits in 48 hours."""
    features = add_aml_features(transactions)
    candidates = features[
        (features["channel"] == "CASH_DEPOSIT")
        & features["is_near_threshold"]
    ].copy()
    flagged_customers = candidates.loc[
        candidates["txn_count_48h"] >= minimum_count, "customer_id"
    ].unique()
    candidates = candidates[candidates["customer_id"].isin(flagged_customers)].copy()
    candidates["rule_flag"] = "STRUCTURING"
    return candidates.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)


def detect_anomalies(
    transactions: pd.DataFrame, contamination: float = 0.01, random_state: int = 42
) -> pd.DataFrame:
    """Flag unusual individual transactions using Isolation Forest.

    The returned DataFrame retains the original fields plus a score and flag.
    Rule-based structuring detection should remain the primary alert for known
    compliance scenarios; this model is useful for unexplained outliers.
    """
    result = transactions.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["hour"] = result["timestamp"].dt.hour
    result["country_risk"] = result["country"].isin(["KY", "PA"]).astype(int)
    result["channel_code"] = pd.factorize(result["channel"])[0]
    model_features = pd.DataFrame(
        {
            "log_amount": np.log1p(result["amount"]),
            "hour": result["hour"],
            "country_risk": result["country_risk"],
            "channel_code": result["channel_code"],
        }
    )
    model = IsolationForest(contamination=contamination, random_state=random_state)
    result["anomaly_score"] = -model.fit(model_features).score_samples(model_features)
    result["ml_flag"] = np.where(model.predict(model_features) == -1, "ML_ANOMALY", "NORMAL")
    return result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
