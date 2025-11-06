import pandas as pd

from utils.explanations import ExplanationEngine
from utils.fraud_engine import FraudEngine


def test_end_to_end_pipeline(data_processor, synthetic_raw_claims, trained_manager):
    raw_df, column_mapping = synthetic_raw_claims
    processed_df = data_processor.prepare_data(raw_df, column_mapping)

    fraud_engine = FraudEngine()
    explanation_engine = ExplanationEngine()

    rule_results = [
        fraud_engine.analyze_single_claim(row.to_dict())
        for _, row in processed_df.iterrows()
    ]

    rule_df = pd.DataFrame(rule_results)
    explained_df = explanation_engine.add_explanations(rule_df)

    combined_df = processed_df.reset_index(drop=True).copy()
    combined_df["risk_score"] = explained_df["risk_score"].astype(float)
    combined_df["triggered_rules"] = explained_df["triggered_rules"].fillna("")

    predictions = trained_manager.predict_batch(combined_df.head(50))

    assert len(predictions) == 50
    assert (predictions >= 0).all() and (predictions <= 1).all()
    assert combined_df["triggered_rules"].str.len().gt(0).mean() > 0.5
