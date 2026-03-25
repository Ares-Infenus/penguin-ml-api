# ADR-004: Deploy — Render

## Status
Accepted

## Context
Need a hosting platform with Docker support, HTTPS, and a free tier for demonstration.

## Decision
Deploy on Render's free tier with Docker runtime.

## Consequences
- Free tier has cold starts (~30-60s after inactivity).
- Automatic HTTPS and health check monitoring.
- Simple Git-based deployment workflow.
