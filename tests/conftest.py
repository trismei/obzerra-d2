import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_processor import DataProcessor
from utils.explanations import ExplanationEngine
from utils.fraud_engine import FraudEngine
from utils.ml_models import MLModelManager


@pytest.fixture(scope="session")
def data_processor() -> DataProcessor:
    return DataProcessor()


@pytest.fixture(scope="session")
def synthetic_raw_claims() -> Tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(2024)
    n_rows = 240

    base_amounts = rng.integers(5000, 120000, size=n_rows)
    # Inject high-risk scenarios in the first 40 rows to guarantee positives
    high_risk_amounts = rng.integers(180000, 320000, size=40)
    base_amounts[:40] = high_risk_amounts

    hours = rng.integers(0, 24, size=n_rows)
    hours[:20] = rng.choice([1, 2, 3, 23], size=20)  # suspicious hours

    witnesses = rng.integers(0, 4, size=n_rows)
    witnesses[:30] = 0  # no witnesses for many high-risk rows

    severities = rng.choice(
        ["Minor Damage", "Major Damage", "Total Loss"], size=n_rows, p=[0.45, 0.35, 0.20]
    )
    severities[:30] = rng.choice(["Major Damage", "Total Loss"], size=30)

    incident_types = rng.choice(
        [
            "Single Vehicle Collision",
            "Multi-vehicle Collision",
            "Parked Car",
            "Vehicle Theft",
        ],
        size=n_rows,
    )
    incident_types[:20] = "Vehicle Theft"

    raw_df = pd.DataFrame(
        {
            "ClaimNumber": [f"CLM{i:04d}" for i in range(n_rows)],
            "ClaimAmount": base_amounts,
            "HourOfDay": hours,
            "DriverAge": rng.integers(21, 70, size=n_rows),
            "IncidentType": incident_types,
            "IncidentSeverity": severities,
            "WitnessCount": witnesses,
            "PoliceReport": rng.choice(["YES", "NO", "UNKNOWN"], size=n_rows, p=[0.6, 0.3, 0.1]),
        }
    )

    column_mapping = {
        "claim_id": "ClaimNumber",
        "total_claim_amount": "ClaimAmount",
        "incident_hour_of_the_day": "HourOfDay",
        "age": "DriverAge",
        "incident_type": "IncidentType",
        "incident_severity": "IncidentSeverity",
        "witnesses": "WitnessCount",
        "police_report_available": "PoliceReport",
    }

    return raw_df, column_mapping


@pytest.fixture(scope="session")
def processed_claims(data_processor: DataProcessor, synthetic_raw_claims):
    raw_df, column_mapping = synthetic_raw_claims
    processed_df = data_processor.prepare_data(raw_df, column_mapping)
    return processed_df


@pytest.fixture(scope="session")
def training_dataframe(processed_claims):
    fraud_engine = FraudEngine()
    explanation_engine = ExplanationEngine()

    rule_results = [
        fraud_engine.analyze_single_claim(row._asdict() if hasattr(row, "_asdict") else row.to_dict())
        for _, row in processed_claims.iterrows()
    ]

    rule_df = pd.DataFrame(rule_results)
    explained_df = explanation_engine.add_explanations(rule_df)

    training_df = processed_claims.reset_index(drop=True).copy()
    training_df["risk_score"] = explained_df["risk_score"].astype(float)
    training_df["triggered_rules"] = explained_df["triggered_rules"].fillna("")

    return training_df


@pytest.fixture(scope="session")
def trained_manager(training_dataframe):
    manager = MLModelManager()
    trained = manager.train_models(training_dataframe)
    if not trained:
        raise RuntimeError("Failed to train ML models for test fixtures")
    return manager


@pytest.fixture(scope="session")
def large_inference_frame(training_dataframe):
    multiplier = math.ceil(5000 / len(training_dataframe))
    expanded_df = pd.concat([training_dataframe] * multiplier, ignore_index=True)
    return expanded_df.head(5000)
