"""Health check and monitoring utilities."""

from datetime import UTC, datetime


def get_health_status(model_loaded: bool, model_version: str | None) -> dict:
    """Generate health check response.

    Args:
        model_loaded: Whether the model pipeline is loaded.
        model_version: Current model version or None.

    Returns:
        Health status dictionary.
    """
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_version": model_version,
        "timestamp": datetime.now(UTC).isoformat(),
    }
