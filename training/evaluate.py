"""Evaluation utilities for the penguin classifier."""

from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(pipeline, X_test, y_test: list[str]) -> dict:
    """Evaluate the trained pipeline and return metrics.

    Args:
        pipeline: Trained sklearn pipeline.
        X_test: Test features DataFrame.
        y_test: True labels.

    Returns:
        Dictionary with accuracy and classification report.
    """
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
    }
