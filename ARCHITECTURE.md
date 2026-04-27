# Architecture Overview

## Objective

The UK Market Intelligence System is a small end-to-end portfolio project that demonstrates practical data engineering, applied machine learning, and lightweight LLM enrichment in a way that is easy to explain and maintain.

## Pipeline Summary

1. Ingestion
   Read UK job posting data from static files or a simple API source and save a raw copy for traceability.
2. Standardization
   Normalize key fields such as company name, location, role title, salary text, posting date, and job description.
3. Storage
   Load curated records into BigQuery tables for downstream analytics and scoring.
4. Benchmarking
   Create explainable salary and demand benchmarks using grouped aggregates and simple statistical or ML features.
5. Company Enrichment
   Add lightweight trust/reputation signals, such as a Trustpilot-style rating snapshot, review volume proxy, or categorical trust tier.
6. API Serving
   Expose scored company and job signals through FastAPI endpoints for downstream consumption.

## Design Principles

- Keep every stage simple enough to finish in a weekend.
- Prefer batch scripts and SQL over orchestration frameworks.
- Use clear, explainable features instead of opaque modeling.
- Treat LLM use as optional enrichment, not a hard dependency.
- Keep secrets in environment variables only.

## Suggested Data Flow

```text
Source files/API
    -> raw data in data/raw
    -> Python standardization in src/ingestion and src/features
    -> BigQuery raw/curated tables
    -> SQL and Python benchmark jobs
    -> company enrichment signals
    -> FastAPI endpoints
```

## Initial Modules

- `src/ingestion`: read and standardize incoming job data
- `src/features`: create features for salary and demand analysis
- `src/enrichment`: add trust/reputation attributes
- `src/pipelines`: coordinate simple batch runs
- `src/api`: expose results through FastAPI
- `sql/`: BigQuery DDL and transformation queries
- `models/artifacts/`: saved model files or benchmark artifacts

## Non-Goals

- Real-time streaming
- Complex orchestration
- Scraping-heavy collection
- Perfect company entity resolution
- Large-scale MLOps infrastructure
