"""End-to-end tests validating predictions against the golden dataset."""

import json
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "golden"


@pytest.mark.e2e
class TestGoldenDataset:
    """Validate predictions against golden dataset."""

    @pytest.fixture(autouse=True)
    def load_golden_data(self):
        """Load golden inputs and expected outputs."""
        inputs_path = GOLDEN_DIR / "golden_inputs.json"
        outputs_path = GOLDEN_DIR / "golden_outputs.json"

        if not inputs_path.exists() or not outputs_path.exists():
            pytest.skip("Golden dataset not found. Run 'make train' first.")

        with open(inputs_path) as f:
            self.golden_inputs = json.load(f)
        with open(outputs_path) as f:
            self.golden_outputs = json.load(f)

    def test_golden_dataset_has_expected_size(self):
        assert len(self.golden_inputs) == 10
        assert len(self.golden_outputs) == 10

    def test_all_predictions_match(self, client):
        for i, (inp, expected) in enumerate(
            zip(self.golden_inputs, self.golden_outputs, strict=False)
        ):
            response = client.post("/predict", json=inp)
            assert response.status_code == 200, f"Sample {i} failed with {response.status_code}"
            data = response.json()
            assert data["prediction"] == expected["prediction"], (
                f"Sample {i}: expected {expected['prediction']}, got {data['prediction']}"
            )

    def test_all_probabilities_within_tolerance(self, client):
        tolerance = 0.01
        for i, (inp, expected) in enumerate(
            zip(self.golden_inputs, self.golden_outputs, strict=False)
        ):
            response = client.post("/predict", json=inp)
            data = response.json()
            for species, expected_prob in expected["probabilities"].items():
                actual_prob = data["probabilities"][species]
                assert abs(actual_prob - expected_prob) <= tolerance, (
                    f"Sample {i}, {species}: expected {expected_prob}, "
                    f"got {actual_prob} (tolerance={tolerance})"
                )

    def test_all_species_represented(self):
        predictions = {out["prediction"] for out in self.golden_outputs}
        assert "Adelie" in predictions
        assert "Chinstrap" in predictions
        assert "Gentoo" in predictions
