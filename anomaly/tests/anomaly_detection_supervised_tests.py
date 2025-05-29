import pandas as pd
from unittest.mock import MagicMock
from src.anomaly_detection_supervised import (
    generate_labels,
    train_supervised_anomaly_model,
    apply_supervised_model
)

def test_generate_labels_creates_labels():
    df = pd.DataFrame({
        "age": [25, 90],
        "weight": [70, 25],
        "height": [175, 160],
        "bmi": [22.0, 60.0]  # One normal, one anomaly
    })

    labeled_df = generate_labels(df)

    assert "label" in labeled_df.columns
    assert set(labeled_df["label"]) <= {0, 1}
    assert len(labeled_df) == 2


def test_train_supervised_anomaly_model_trains_xgb():
    df = pd.DataFrame({
        "age": [25, 90],
        "weight": [70, 25],
        "height": [175, 160],
        "bmi": [22.0, 60.0]
    })

    df_labeled = generate_labels(df)
    model, scaler = train_supervised_anomaly_model(df_labeled)

    assert hasattr(model, "predict_proba")
    assert hasattr(scaler, "transform")


def test_apply_supervised_model_with_mocked_model_and_scaler():
    df = pd.DataFrame({
        "age": [25],
        "weight": [70],
        "height": [175],
        "bmi": [22.0]
    })

    # Mock the model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.1, 0.9]]  # Mock anomaly score 0.9

    # Mock the scaler
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = df[["age", "weight", "height", "bmi"]].values

    result_df = apply_supervised_model(df, mock_model, mock_scaler)

    assert "ai_anomaly_score" in result_df.columns
    assert result_df["ai_anomaly_score"].iloc[0] == 0.9
