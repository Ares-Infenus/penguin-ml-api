"""Smoke test against a deployed URL."""

import sys

import httpx


def smoke_test(base_url: str) -> None:
    """Run smoke tests against the deployed API.

    Args:
        base_url: Base URL of the deployed API (e.g., https://app.onrender.com).
    """
    base_url = base_url.rstrip("/")
    print(f"Running smoke tests against: {base_url}")

    # Test health
    print("\n[1/2] Testing GET /health...")
    r = httpx.get(f"{base_url}/health", timeout=60)
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    data = r.json()
    assert data["status"] == "healthy"
    print(f"  Status: {data['status']}, Model: {data['model_version']}")

    # Test prediction
    print("\n[2/2] Testing POST /predict...")
    payload = {
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "sex": "male",
        "island": "Torgersen",
    }
    r = httpx.post(f"{base_url}/predict", json=payload, timeout=60)
    assert r.status_code == 200, f"Predict failed: {r.status_code}"
    data = r.json()
    assert data["prediction"] in ["Adelie", "Chinstrap", "Gentoo"]
    print(f"  Prediction: {data['prediction']}")
    print(f"  Probabilities: {data['probabilities']}")

    print("\nAll smoke tests passed!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_live.py <URL>")
        sys.exit(1)
    smoke_test(sys.argv[1])
