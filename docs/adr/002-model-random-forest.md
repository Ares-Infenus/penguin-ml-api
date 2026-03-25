# ADR-002: Model — Random Forest + sklearn Pipeline

## Status
Accepted

## Context
Need a classifier that handles mixed features well and integrates easily into a serializable pipeline.

## Decision
Use RandomForestClassifier wrapped in an sklearn Pipeline with ColumnTransformer for preprocessing.

## Consequences
- Single serializable artifact includes both preprocessing and model.
- Good baseline accuracy without extensive tuning.
- No GPU or heavy framework dependencies.
