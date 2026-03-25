# ADR-003: Framework — FastAPI

## Status
Accepted

## Context
Need a web framework for serving ML predictions with input validation and auto-generated documentation.

## Decision
Use FastAPI with Pydantic v2 for validation and automatic OpenAPI spec generation.

## Consequences
- Built-in request validation reduces boilerplate.
- Swagger UI available at /docs out of the box.
- Async support for future scalability.
