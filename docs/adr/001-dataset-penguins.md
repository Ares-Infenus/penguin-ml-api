# ADR-001: Dataset — Palmer Penguins

## Status
Accepted

## Context
We need a dataset for a multi-class classification demo that exercises a realistic ML pipeline (mixed feature types, missing values, preprocessing).

## Decision
Use the Palmer Penguins dataset (344 samples, 3 species, 6 features: 4 numeric + 2 categorical).

## Consequences
- Requires handling of NaN values and mixed types.
- Small enough for fast training, large enough for meaningful evaluation.
- Well-documented and widely used in ML education.
