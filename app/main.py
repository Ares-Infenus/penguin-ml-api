"""FastAPI application for penguin species prediction."""

import json
import logging
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException

from app.config import settings
from app.monitoring import get_health_status
from app.predict import get_prediction
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PenguinFeatures,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

# Application state
app_state: dict = {
    "pipeline": None,
    "metadata": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    pipeline_path = settings.model_path_resolved
    metadata_path = settings.metadata_path_resolved

    try:
        app_state["pipeline"] = joblib.load(pipeline_path)
        with open(metadata_path) as f:
            app_state["metadata"] = json.load(f)
        logger.info("Model loaded: version=%s", app_state["metadata"].get("model_version"))
    except FileNotFoundError as e:
        logger.error("Model artifacts not found: %s", e)
        raise

    yield

    app_state["pipeline"] = None
    app_state["metadata"] = None
    logger.info("Application shutdown, model unloaded.")


app = FastAPI(
    title="Penguin Species Predictor",
    description="API REST for classifying penguin species from morphological measurements.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    model_loaded = app_state["pipeline"] is not None
    model_version = app_state["metadata"].get("model_version") if app_state["metadata"] else None
    return get_health_status(model_loaded, model_version)


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PenguinFeatures):
    """Predict penguin species from morphological features."""
    if app_state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_version = app_state["metadata"].get("model_version", "unknown")

    result = get_prediction(features, app_state["pipeline"], model_version)

    logger.info(
        "Prediction: input=%s, prediction=%s",
        features.model_dump(),
        result["prediction"],
    )

    return result


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata."""
    if app_state["metadata"] is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")

    metadata = app_state["metadata"]
    return {
        "model_version": metadata["model_version"],
        "model_type": metadata["model_type"],
        "training_date": metadata["training_date"],
        "accuracy": metadata["accuracy"],
        "features": metadata["features"],
        "target_classes": metadata["target_classes"],
        "pipeline_sha256": metadata["pipeline_sha256"],
    }
