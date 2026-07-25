"""Load and filter transaction data safely."""

from pathlib import Path

import pandas as pd


# Expected location:
# outputs/
# ├── aml_transactions.csv
# └── src/tools/filter_tool.py
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "aml_transactions.csv"


def load_transactions(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the transaction dataset from a CSV or Excel file."""
    path = Path(data_path)

    if not path.exists():
        # Allows aml_transactions.xlsx when a CSV path was supplied.
        excel_path = path.with_suffix(".xlsx")
        if excel_path.exists():
            path = excel_path
        else:
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                f"Also checked: {excel_path}"
            )

    if path.suffix.lower() in {".xlsx", ".xls"}:
        transactions = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        transactions = pd.read_csv(path)
    else:
        raise ValueError("Supported dataset formats are .csv, .xlsx, and .xls.")

    required_columns = {
        "transaction_id",
        "customer_id",
        "timestamp",
        "amount",
        "channel",
        "country",
    }
    missing_columns = required_columns - set(transactions.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
    transactions["amount"] = pd.to_numeric(transactions["amount"])

    return transactions.sort_values("timestamp").reset_index(drop=True)


def filter_transactions(
    transactions: pd.DataFrame,
    customer_id: str | None = None,
    date_start: str | pd.Timestamp | None = None,
    date_end: str | pd.Timestamp | None = None,
    country: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
) -> pd.DataFrame:
    """Return transactions that match every supplied filter."""
    if min_amount is not None and max_amount is not None and min_amount > max_amount:
        raise ValueError("min_amount cannot be greater than max_amount.")

    result = transactions.copy()

    if customer_id is not None:
        result = result[result["customer_id"] == customer_id]
    if date_start is not None:
        result = result[result["timestamp"] >= pd.Timestamp(date_start)]
    if date_end is not None:
        result = result[result["timestamp"] <= pd.Timestamp(date_end)]
    if country is not None:
        result = result[result["country"].str.upper() == country.upper()]
    if min_amount is not None:
        result = result[result["amount"] >= min_amount]
    if max_amount is not None:
        result = result[result["amount"] <= max_amount]

    return result.sort_values("timestamp").reset_index(drop=True)