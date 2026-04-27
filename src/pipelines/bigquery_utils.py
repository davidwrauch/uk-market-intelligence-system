"""Small BigQuery helpers for local pipeline scripts.

This module intentionally keeps the interface simple:
- create a BigQuery client from environment variables
- load a pandas DataFrame into a BigQuery table
- run a SQL file from the local sql/ directory
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery


def _load_settings() -> tuple[str, str, str]:
    """Load required BigQuery settings from environment variables."""
    load_dotenv()

    project_id = os.getenv("GCP_PROJECT_ID")
    dataset = os.getenv("BIGQUERY_DATASET")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not project_id:
        raise ValueError("Missing GCP_PROJECT_ID environment variable.")
    if not dataset:
        raise ValueError("Missing BIGQUERY_DATASET environment variable.")
    if not credentials_path:
        raise ValueError("Missing GOOGLE_APPLICATION_CREDENTIALS environment variable.")

    return project_id, dataset, credentials_path


def _resolve_table_id(table_id: str, project_id: str, dataset: str) -> str:
    """Allow callers to pass either a short table name or a full table id."""
    if "." in table_id:
        return table_id
    return f"{project_id}.{dataset}.{table_id}"


def get_bigquery_client() -> bigquery.Client:
    """Create and return a BigQuery client using environment configuration."""
    project_id, _, credentials_path = _load_settings()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    return bigquery.Client(project=project_id)


def load_dataframe_to_bigquery(
    df,
    table_id: str,
    write_disposition: str = "WRITE_APPEND",
) -> None:
    """Load a pandas DataFrame into a BigQuery table.

    The write disposition defaults to append, but can be overridden for simple
    one-off refresh jobs.
    """
    project_id, dataset, _ = _load_settings()
    client = get_bigquery_client()
    full_table_id = _resolve_table_id(table_id, project_id, dataset)

    job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
    load_job = client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
    load_job.result()

    print(f"Loaded {len(df)} rows into {full_table_id}")


def run_sql_file(sql_path: str | Path) -> None:
    """Read a SQL file, format project/dataset placeholders, and execute it."""
    project_id, dataset, _ = _load_settings()
    client = get_bigquery_client()

    sql_file = Path(sql_path)
    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_file}")

    query_text = sql_file.read_text(encoding="utf-8").format(
        project_id=project_id,
        dataset=dataset,
    )

    query_job = client.query(query_text)
    query_job.result()

    print(f"Executed SQL file: {sql_file}")
