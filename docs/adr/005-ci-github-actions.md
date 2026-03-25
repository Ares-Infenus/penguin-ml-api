# ADR-005: CI — GitHub Actions

## Status
Accepted

## Context
Need automated linting, testing, and build verification on every push and PR.

## Decision
Use GitHub Actions with a 3-stage pipeline: lint → test → docker build.

## Consequences
- Free for public repositories.
- Native GitHub integration for PR checks.
- Marketplace ecosystem for reusable actions.
