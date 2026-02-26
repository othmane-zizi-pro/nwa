from datetime import datetime

import pandas as pd

from src.features import (
    calculate_clv_target,
    calculate_rfm_features,
    split_data_temporal,
)


def _sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Customer ID": [1, 1, 2, 2, 3],
            "Invoice": ["A1", "A2", "B1", "B2", "C1"],
            "InvoiceDate": pd.to_datetime(
                [
                    "2010-01-01",
                    "2010-03-01",
                    "2010-02-01",
                    "2010-07-01",
                    "2010-08-01",
                ]
            ),
            "TotalAmount": [100.0, 200.0, 50.0, 70.0, 30.0],
        }
    )


def test_calculate_rfm_features_basic():
    df = _sample_transactions()
    reference_date = datetime(2010, 9, 1)
    rfm = calculate_rfm_features(df, reference_date=reference_date)

    assert set(["Customer ID", "Recency", "Frequency", "Monetary"]).issubset(rfm.columns)
    row_c1 = rfm[rfm["Customer ID"] == 1].iloc[0]
    assert row_c1["Frequency"] == 2
    assert row_c1["Monetary"] == 300.0


def test_split_data_temporal_has_non_empty_windows():
    df = _sample_transactions()
    obs, pred, _, _ = split_data_temporal(df, observation_months=6, prediction_months=2)
    assert len(obs) > 0
    assert len(pred) > 0


def test_calculate_clv_target_includes_zero_customers():
    df = _sample_transactions()
    obs = df[df["InvoiceDate"] < pd.Timestamp("2010-06-01")]
    pred = df[df["InvoiceDate"] >= pd.Timestamp("2010-06-01")]

    clv = calculate_clv_target(obs, pred)
    # Customer 1 has no purchase in prediction window -> CLV should be zero.
    row = clv[clv["Customer ID"] == 1].iloc[0]
    assert row["CLV"] == 0
