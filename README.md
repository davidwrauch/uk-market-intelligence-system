# UK Market Intelligence System

Small production-style ML + LLM portfolio project focused on UK job market signals. The system will ingest UK job postings, store curated data in BigQuery, generate salary and demand benchmarks, lightly enrich company records with trust/reputation signals, and expose scored outputs through FastAPI.

## Goals

- Build a weekend-sized, explainable data product for applied data science interviews.
- Show practical use of Python, SQL, pandas, scikit-learn, BigQuery, and FastAPI.
- Keep the implementation simple, robust, and easy to discuss in interviews.

## Planned Scope

- Ingest sample UK job posting data from files or APIs.
- Standardize fields needed for analytics and downstream scoring.
- Store raw and modeled tables in BigQuery.
- Create simple salary and demand benchmarks with transparent logic.
- Add lightweight company trust/reputation enrichment.
- Serve job and company signals through a FastAPI API.

## Tech Stack

- Python
- pandas
- scikit-learn
- BigQuery
- SQL
- FastAPI
- Docker
- GitHub Actions
- Optional LLM structured extraction

## Repository Structure

```text
.
|-- app/
|-- src/
|   |-- api/
|   |-- enrichment/
|   |-- features/
|   |-- ingestion/
|   `-- pipelines/
|-- sql/
|-- models/
|   `-- artifacts/
|-- data/
|   |-- raw/
|   `-- processed/
`-- .github/
    `-- workflows/
```

## Setup

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env` and fill in required values.
4. Authenticate to Google Cloud before running BigQuery-related steps.

## Development Plan

1. Add a minimal ingestion pipeline for job posting source files.
2. Create BigQuery table schemas and SQL transforms.
3. Implement benchmark generation for salary and demand.
4. Add lightweight company enrichment and scoring.
5. Expose outputs with FastAPI.
6. Add Docker and GitHub Actions for local/dev workflow.

## Status

Project scaffold created. Pipeline implementation is intentionally not started yet.
