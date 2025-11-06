from utils.feature_explainer import FeatureExplainer


def test_feature_explainer_generates_contributions(trained_manager, training_dataframe):
    explainer = FeatureExplainer()
    explainer.fit(trained_manager, training_dataframe[trained_manager.feature_columns])

    sample_row = training_dataframe.iloc[0].to_dict()
    probability = trained_manager.predict_single(training_dataframe.iloc[0])
    explanation = explainer.explain_prediction(sample_row, probability)

    assert "contributions" in explanation
    assert explanation["contributions"]  # non-empty
    top_positive = explanation["top_positive"]
    top_negative = explanation["top_negative"]

    # Ensure formatted output is human readable
    if top_positive:
        assert all("feature" in item and "contribution" in item for item in top_positive)
    if top_negative:
        assert all("feature" in item and "contribution" in item for item in top_negative)
