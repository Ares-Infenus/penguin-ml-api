"""Unit tests for prediction logic."""

import joblib
import pytest

from app.config import settings
from app.predict import get_prediction
from app.schemas import PenguinFeatures


@pytest.fixture(scope="module")
def pipeline():
    """Load the trained pipeline for unit tests."""
    path = settings.model_path_resolved
    if not path.exists():
        pytest.skip("Model not found. Run training first.")
    return joblib.load(path)


@pytest.mark.unit
class TestGetPrediction:
    """Tests for the get_prediction function."""

    def test_returns_expected_keys(self, valid_input, pipeline):
        features = PenguinFeatures(**valid_input)
        result = get_prediction(features, pipeline, "1.0.0")

        assert "prediction" in result
        assert "probabilities" in result
        assert "model_version" in result

    def test_prediction_is_valid_class(self, valid_input, pipeline):
        features = PenguinFeatures(**valid_input)
        result = get_prediction(features, pipeline, "1.0.0")

        assert result["prediction"] in ["Adelie", "Chinstrap", "Gentoo"]

    def test_probabilities_sum_to_one(self, valid_input, pipeline):
        features = PenguinFeatures(**valid_input)
        result = get_prediction(features, pipeline, "1.0.0")

        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_model_version_passed_through(self, valid_input, pipeline):
        features = PenguinFeatures(**valid_input)
        result = get_prediction(features, pipeline, "2.0.0")

        assert result["model_version"] == "2.0.0"
