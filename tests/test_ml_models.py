import numpy as np
import pytest


def test_predict_batch_returns_probabilities(trained_manager, training_dataframe):
    sample = training_dataframe.head(25)
    predictions = trained_manager.predict_batch(sample)

    assert len(predictions) == len(sample)
    assert np.all((predictions >= 0) & (predictions <= 1))


def test_individual_models_produce_outputs(trained_manager, training_dataframe):
    feature_cols = trained_manager.feature_columns
    feature_frame = training_dataframe[feature_cols]
    scaled = trained_manager.scaler.transform(feature_frame)

    lr_probs = trained_manager.logistic_model.predict_proba(scaled)[:, 1]
    rf_probs = trained_manager.rf_model.predict_proba(scaled)[:, 1]

    assert lr_probs.shape[0] == len(feature_frame)
    assert rf_probs.shape[0] == len(feature_frame)
    # Ensure the two models learn different decision surfaces
    assert not np.allclose(lr_probs, rf_probs)


def test_predict_single_consistency(trained_manager, training_dataframe):
    row = training_dataframe.iloc[0]
    prob_single = trained_manager.predict_single(row)
    batch_prob = trained_manager.predict_batch(training_dataframe.head(1))[0]

    assert pytest.approx(prob_single, rel=1e-6) == batch_prob
