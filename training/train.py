"""Reproducible training script for the penguin species classifier."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from training.evaluate import evaluate_model

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    """Load training configuration from YAML."""
    config_path = ROOT / "training" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load Palmer Penguins dataset and clean it.

    Args:
        config: Training configuration dict.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    try:
        import seaborn as sns

        df = sns.load_dataset("penguins")
    except Exception:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
        df = pd.read_csv(url)

    df = df.dropna().reset_index(drop=True)

    feature_cols = config["features"]["numeric"] + config["features"]["categorical"]
    X = df[feature_cols]
    y = df[config["target"]]

    return X, y


def build_pipeline(config: dict) -> Pipeline:
    """Build sklearn pipeline with preprocessing and classifier.

    Args:
        config: Training configuration dict.

    Returns:
        Untrained sklearn Pipeline.
    """
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="if_binary", sparse_output=False), categorical_features),
        ]
    )

    model_params = config["model"]["params"]
    classifier = RandomForestClassifier(**model_params)

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_golden_dataset(pipeline, X: pd.DataFrame, y: pd.Series, config: dict) -> None:
    """Generate and save golden dataset for regression testing.

    Selects 10 representative samples covering all species, sexes, and islands.
    """
    golden_dir = ROOT / "data" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    # Select representative samples: stratified by species
    indices = []
    for species in y.unique():
        species_mask = y == species
        species_indices = y[species_mask].index.tolist()
        # Take first 3-4 per species to get ~10 total
        n_samples = 4 if species == y.unique()[0] else 3
        indices.extend(species_indices[:n_samples])

    golden_X = X.iloc[indices]
    golden_y = y.iloc[indices]

    # Generate predictions
    predictions = pipeline.predict(golden_X)
    probabilities = pipeline.predict_proba(golden_X)
    class_names = pipeline.classes_.tolist()

    # Build golden inputs (lowercase sex to match API schema)
    golden_inputs = []
    for _, row in golden_X.iterrows():
        entry = row.to_dict()
        entry["sex"] = entry["sex"].lower()
        golden_inputs.append(entry)

    # Build golden outputs
    golden_outputs = []
    for i, pred in enumerate(predictions):
        proba_dict = {
            class_names[j]: round(float(probabilities[i][j]), 4) for j in range(len(class_names))
        }
        golden_outputs.append(
            {
                "prediction": pred,
                "probabilities": proba_dict,
            }
        )

    with open(golden_dir / "golden_inputs.json", "w") as f:
        json.dump(golden_inputs, f, indent=2)

    with open(golden_dir / "golden_outputs.json", "w") as f:
        json.dump(golden_outputs, f, indent=2)

    print(f"Golden dataset saved: {len(golden_inputs)} samples")
    print(f"  Species distribution: {golden_y.value_counts().to_dict()}")


def train() -> None:
    """Execute the full training pipeline."""
    config = load_config()
    print("=" * 60)
    print("Penguin Species Classifier — Training Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading data...")
    X, y = load_data(config)
    print(f"  Samples: {len(X)}, Features: {list(X.columns)}")
    print(f"  Classes: {y.unique().tolist()}")

    # Split data
    print("\n[2/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Build pipeline
    print("\n[3/6] Building pipeline...")
    pipeline = build_pipeline(config)
    print(f"  Model: {config['model']['type']}")
    print(f"  Params: {config['model']['params']}")

    # Train
    print("\n[4/6] Training...")
    pipeline.fit(X_train, y_train)
    print("  Training complete.")

    # Evaluate
    print("\n[5/6] Evaluating...")
    metrics = evaluate_model(pipeline, X_test, y_test)
    print(f"  Accuracy: {metrics['accuracy']}")

    min_accuracy = config["thresholds"]["min_accuracy"]
    if metrics["accuracy"] < min_accuracy:
        raise ValueError(f"Accuracy {metrics['accuracy']} below threshold {min_accuracy}")
    print(f"  Threshold ({min_accuracy}) passed!")

    # Save artifacts
    print("\n[6/6] Saving artifacts...")
    model_dir = ROOT / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = ROOT / config["output"]["pipeline_path"]
    metadata_path = ROOT / config["output"]["metadata_path"]

    joblib.dump(pipeline, pipeline_path)
    pipeline_sha256 = compute_sha256(pipeline_path)

    metadata = {
        "model_version": config["output"]["version"],
        "model_type": config["model"]["type"],
        "training_date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "accuracy": metrics["accuracy"],
        "classification_report": metrics["classification_report"],
        "features": config["features"]["numeric"] + config["features"]["categorical"],
        "target_classes": sorted(y.unique().tolist()),
        "pipeline_sha256": pipeline_sha256,
        "data_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "hyperparameters": config["model"]["params"],
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Pipeline: {pipeline_path} (SHA256: {pipeline_sha256[:12]}...)")
    print(f"  Metadata: {metadata_path}")

    # Generate golden dataset
    print("\n[Bonus] Generating golden dataset...")
    save_golden_dataset(pipeline, X, y, config)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()
