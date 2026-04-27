"""Run a simple local-to-BigQuery pipeline for raw UK job data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipelines.bigquery_utils import load_dataframe_to_bigquery, run_sql_file


def get_repo_root() -> Path:
    """Resolve the repository root from this file's location."""
    return Path(__file__).resolve().parents[2]


def main() -> None:
    """Load raw jobs into BigQuery and build the feature table."""
    repo_root = get_repo_root()
    raw_jobs_path = repo_root / "data" / "raw" / "raw_jobs.csv"
    feature_sql_path = repo_root / "sql" / "create_feature_jobs.sql"

    if not raw_jobs_path.exists():
        raise FileNotFoundError(f"Raw jobs file not found: {raw_jobs_path}")

    raw_jobs_df = pd.read_csv(raw_jobs_path)

    print(f"Rows loaded from CSV: {len(raw_jobs_df)}")

    load_dataframe_to_bigquery(
        df=raw_jobs_df,
        table_id="raw_jobs",
        write_disposition="WRITE_TRUNCATE",
    )
    print("Table written: raw_jobs")

    run_sql_file(feature_sql_path)
    print("Table written: feature_jobs")

    print("BigQuery pipeline completed successfully.")


if __name__ == "__main__":
    main()
