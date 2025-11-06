import time

from utils.feature_explainer import FeatureExplainer


PERFORMANCE_BUDGET_SECONDS = 30.0


def test_batch_inference_under_budget(trained_manager, large_inference_frame):
    start = time.perf_counter()
    predictions = trained_manager.predict_batch(large_inference_frame)
    duration = time.perf_counter() - start

    assert len(predictions) == len(large_inference_frame)
    assert duration < PERFORMANCE_BUDGET_SECONDS


def test_shap_generation_under_budget(trained_manager, large_inference_frame):
    explainer = FeatureExplainer()
    explainer.fit(trained_manager, large_inference_frame[trained_manager.feature_columns])

    sample_rows = large_inference_frame.head(25)
    sample_probs = trained_manager.predict_batch(sample_rows)

    start = time.perf_counter()
    for offset, (_, row) in enumerate(sample_rows.iterrows()):
        explainer.explain_prediction(row.to_dict(), sample_probs[offset])
    duration = time.perf_counter() - start

    assert duration < PERFORMANCE_BUDGET_SECONDS
