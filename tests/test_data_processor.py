import pandas as pd
import pytest

from utils.data_processor import DataProcessor


def test_prepare_data_handles_missing_and_engineers_features(data_processor):
    raw_df = pd.DataFrame(
        {
            "ClaimNumber": ["A1", "A2"],
            "ClaimAmount": [10000, 250000],
            "HourOfDay": [8, 2],
            "DriverAge": [30, 22],
            "WitnessCount": [1, 0],
        }
    )

    mapping = {
        "claim_id": "ClaimNumber",
        "total_claim_amount": "ClaimAmount",
        "incident_hour_of_the_day": "HourOfDay",
        "age": "DriverAge",
        "witnesses": "WitnessCount",
    }

    processed = data_processor.prepare_data(raw_df, mapping)

    # Required engineered features should exist
    expected_columns = {
        "log_claim_amount",
        "amount_category",
        "is_round_amount",
        "hour_category",
        "unusual_hour",
    }
    assert expected_columns.issubset(set(processed.columns))

    # Missing optional columns get defaults without raising errors
    assert processed.loc[0, "age"] == 30
    assert processed.loc[1, "unusual_hour"] in (0, 1)


def test_prepare_data_raises_without_amount():
    processor = DataProcessor()
    raw_df = pd.DataFrame({"ClaimNumber": ["A1"], "HourOfDay": [12]})

    mapping = {
        "claim_id": "ClaimNumber",
        "incident_hour_of_the_day": "HourOfDay",
    }

    with pytest.raises(Exception) as exc:
        processor.prepare_data(raw_df, mapping)

    assert "Required column" in str(exc.value)
