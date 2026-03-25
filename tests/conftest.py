"""Shared fixtures for tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """FastAPI test client with model loaded."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_input():
    """Valid penguin features for testing."""
    return {
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "sex": "male",
        "island": "Torgersen",
    }
