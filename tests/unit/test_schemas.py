"""Unit tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from app.schemas import PenguinFeatures


@pytest.mark.unit
class TestPenguinFeatures:
    """Tests for PenguinFeatures schema validation."""

    def test_valid_input_creates_schema(self, valid_input):
        features = PenguinFeatures(**valid_input)
        assert features.bill_length_mm == 39.1
        assert features.sex.value == "male"
        assert features.island.value == "Torgersen"

    def test_valid_boundary_min(self):
        features = PenguinFeatures(
            bill_length_mm=25.0,
            bill_depth_mm=12.0,
            flipper_length_mm=170.0,
            body_mass_g=2500.0,
            sex="female",
            island="Dream",
        )
        assert features.bill_length_mm == 25.0

    def test_valid_boundary_max(self):
        features = PenguinFeatures(
            bill_length_mm=65.0,
            bill_depth_mm=22.0,
            flipper_length_mm=240.0,
            body_mass_g=6500.0,
            sex="male",
            island="Biscoe",
        )
        assert features.bill_length_mm == 65.0

    @pytest.mark.parametrize(
        "field,value",
        [
            ("bill_length_mm", 24.9),
            ("bill_length_mm", 65.1),
            ("bill_depth_mm", 11.9),
            ("bill_depth_mm", 22.1),
            ("flipper_length_mm", 169.9),
            ("flipper_length_mm", 240.1),
            ("body_mass_g", 2499.9),
            ("body_mass_g", 6500.1),
        ],
    )
    def test_numeric_out_of_range_raises_error(self, valid_input, field, value):
        valid_input[field] = value
        with pytest.raises(ValidationError):
            PenguinFeatures(**valid_input)

    def test_invalid_sex_raises_error(self, valid_input):
        valid_input["sex"] = "unknown"
        with pytest.raises(ValidationError):
            PenguinFeatures(**valid_input)

    def test_invalid_island_raises_error(self, valid_input):
        valid_input["island"] = "Antarctica"
        with pytest.raises(ValidationError):
            PenguinFeatures(**valid_input)

    def test_missing_field_raises_error(self):
        with pytest.raises(ValidationError):
            PenguinFeatures(
                bill_length_mm=39.1,
                bill_depth_mm=18.7,
                # missing flipper_length_mm and others
            )

    def test_wrong_type_raises_error(self, valid_input):
        valid_input["bill_length_mm"] = "not_a_number"
        with pytest.raises(ValidationError):
            PenguinFeatures(**valid_input)
