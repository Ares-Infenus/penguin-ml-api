"""Prediction logic for the penguin classifier."""

import pandas as pd

from app.schemas import PenguinFeatures


def get_prediction(features: PenguinFeatures, pipeline, model_version: str) -> dict:
    """Run prediction using the loaded pipeline.

    Args:
        features: Validated input features.
        pipeline: Trained sklearn pipeline.
        model_version: Current model version string.

    Returns:
        Dictionary with prediction, probabilities, and model version.
    """
    input_df = pd.DataFrame(
        [
            {
                "bill_length_mm": features.bill_length_mm,
                "bill_depth_mm": features.bill_depth_mm,
                "flipper_length_mm": features.flipper_length_mm,
                "body_mass_g": features.body_mass_g,
                "sex": features.sex.value.capitalize(),
                "island": features.island.value,
            }
        ]
    )

    prediction = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]
    class_names = pipeline.classes_.tolist()

    proba_dict = {
        class_names[i]: round(float(probabilities[i]), 4) for i in range(len(class_names))
    }

    return {
        "prediction": prediction,
        "probabilities": proba_dict,
        "model_version": model_version,
    }
