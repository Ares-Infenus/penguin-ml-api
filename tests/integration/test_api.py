"""Integration tests for API endpoints."""

import pytest


@pytest.mark.integration
class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_valid_input_returns_200(self, client, valid_input):
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probabilities" in data
        assert "model_version" in data
        assert data["prediction"] in ["Adelie", "Chinstrap", "Gentoo"]

    def test_missing_field_returns_422(self, client):
        response = client.post("/predict", json={"bill_length_mm": 39.1})
        assert response.status_code == 422

    def test_out_of_range_returns_422(self, client, valid_input):
        valid_input["bill_length_mm"] = 100.0
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422

    def test_wrong_type_returns_422(self, client, valid_input):
        valid_input["bill_length_mm"] = "not_a_number"
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422

    def test_invalid_sex_returns_422(self, client, valid_input):
        valid_input["sex"] = "unknown"
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422

    def test_invalid_island_returns_422(self, client, valid_input):
        valid_input["island"] = "Mars"
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422

    def test_probabilities_sum_to_one(self, client, valid_input):
        response = client.post("/predict", json=valid_input)
        data = response.json()
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 0.01


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "timestamp" in data

    def test_health_includes_model_version(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["model_version"] is not None


@pytest.mark.integration
class TestModelInfoEndpoint:
    """Tests for GET /model/info."""

    def test_model_info_returns_200(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data
        assert "model_type" in data
        assert "accuracy" in data
        assert "features" in data
        assert "target_classes" in data

    def test_model_info_has_correct_classes(self, client):
        response = client.get("/model/info")
        data = response.json()
        assert sorted(data["target_classes"]) == ["Adelie", "Chinstrap", "Gentoo"]
