# ADR-006: Serialization — joblib

## Status
Accepted

## Context
Need to serialize the trained sklearn pipeline to disk for loading at API startup.

## Decision
Use joblib for pipeline serialization.

## Consequences
- More efficient than pickle for objects containing numpy arrays.
- Standard approach in sklearn documentation and ecosystem.
- Single file artifact simplifies deployment.
