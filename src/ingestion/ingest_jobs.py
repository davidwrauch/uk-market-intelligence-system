"""Fetch UK job postings from Adzuna and save them locally as a CSV file.

This ingestion step aims for broader but still manageable UK coverage by querying:
- a wider list of data/AI/analytics search terms
- targeted city searches for London, Edinburgh, and Glasgow
- a UK-wide search with no location filter
- a configurable number of pages and results per page from environment variables
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv


SEARCH_TERMS = [
    "data scientist",
    "senior data scientist",
    "machine learning engineer",
    "ml engineer",
    "applied scientist",
    "data analyst",
    "senior data analyst",
    "product analyst",
    "analytics engineer",
    "business intelligence",
    "risk analyst",
    "fraud analyst",
    "pricing analyst",
    "decision scientist",
    "ai engineer",
    "applied ai",
    "machine learning",
]

QUERY_GROUPS = [
    {
        "search_location": "Edinburgh",
        "where": "Edinburgh",
        "location_filter_applied": True,
    },
    {
        "search_location": "Glasgow",
        "where": "Glasgow",
        "location_filter_applied": True,
    },
    {
        "search_location": "London",
        "where": "London",
        "location_filter_applied": True,
    },
    {
        "search_location": "UK-wide",
        "where": None,
        "location_filter_applied": False,
    },
]

DEFAULT_PAGES = 5
DEFAULT_RESULTS_PER_PAGE = 50
DEFAULT_SLEEP_SECONDS = 1.0
ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs/gb/search"

COLUMNS = [
    "ingested_at",
    "job_id",
    "search_term",
    "search_location",
    "location_filter_applied",
    "title",
    "company",
    "location",
    "category",
    "description",
    "redirect_url",
    "created",
    "salary_min",
    "salary_max",
]


def parse_args() -> argparse.Namespace:
    """Parse command line options for a small amount of configurability."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest UK jobs from the Adzuna API.")
    parser.add_argument(
        "--pages",
        type=int,
        default=int(os.getenv("ADZUNA_PAGES_PER_QUERY", DEFAULT_PAGES)),
        help="Number of result pages to fetch for each search term and location.",
    )
    parser.add_argument(
        "--results-per-page",
        type=int,
        default=int(os.getenv("ADZUNA_RESULTS_PER_PAGE", DEFAULT_RESULTS_PER_PAGE)),
        help="Number of jobs to request per API call.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Seconds to sleep between API requests for basic rate limiting.",
    )
    return parser.parse_args()


def get_repo_root() -> Path:
    """Resolve the repository root from this file's location."""
    return Path(__file__).resolve().parents[2]


def get_output_path() -> Path:
    """Return the local CSV output path and ensure its folder exists."""
    output_path = get_repo_root() / "data" / "raw" / "raw_jobs.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def load_credentials() -> tuple[str, str]:
    """Load Adzuna credentials from environment variables."""
    load_dotenv()

    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        raise ValueError(
            "Missing Adzuna credentials. Set ADZUNA_APP_ID and ADZUNA_APP_KEY in your environment."
        )

    return app_id, app_key


def fetch_jobs_page(
    client: httpx.Client,
    app_id: str,
    app_key: str,
    search_term: str,
    location_filter: str | None,
    page: int,
    results_per_page: int,
) -> list[dict[str, Any]]:
    """Fetch one page of job results from Adzuna."""
    url = f"{ADZUNA_BASE_URL}/{page}"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": search_term,
        "results_per_page": results_per_page,
        "content-type": "application/json",
    }
    if location_filter:
        params["where"] = location_filter

    response = client.get(url, params=params)
    response.raise_for_status()

    payload = response.json()
    results = payload.get("results", [])

    if not isinstance(results, list):
        return []

    return results


def normalize_job(
    job: dict[str, Any],
    search_term: str,
    search_location: str,
    location_filter_applied: bool,
    ingested_at: str,
) -> dict[str, Any]:
    """Flatten one Adzuna job record into our local raw schema."""
    company = (job.get("company") or {}).get("display_name")
    location = (job.get("location") or {}).get("display_name")
    category = (job.get("category") or {}).get("label")

    return {
        "ingested_at": ingested_at,
        "job_id": job.get("id"),
        "search_term": search_term,
        "search_location": search_location,
        "location_filter_applied": location_filter_applied,
        "title": job.get("title"),
        "company": company,
        "location": location,
        "category": category,
        "description": job.get("description"),
        "redirect_url": job.get("redirect_url"),
        "created": job.get("created"),
        "salary_min": job.get("salary_min"),
        "salary_max": job.get("salary_max"),
    }


def build_jobs_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a DataFrame with a fixed column order, even when no rows are returned."""
    return pd.DataFrame(records, columns=COLUMNS)


def ingest_jobs(
    pages: int = DEFAULT_PAGES,
    results_per_page: int = DEFAULT_RESULTS_PER_PAGE,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> pd.DataFrame:
    """Fetch Adzuna job results for all configured search terms and query groups."""
    if pages < 1:
        raise ValueError("pages must be at least 1")
    if results_per_page < 1:
        raise ValueError("results_per_page must be at least 1")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds cannot be negative")

    app_id, app_key = load_credentials()
    ingested_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []

    # A single client keeps the code simple and reuses the connection across requests.
    with httpx.Client(timeout=30.0, headers={"Accept": "application/json"}) as client:
        for search_term in SEARCH_TERMS:
            for query_group in QUERY_GROUPS:
                search_location = query_group["search_location"]
                location_filter = query_group["where"]
                location_filter_applied = query_group["location_filter_applied"]
                combo_rows_fetched = 0

                for page in range(1, pages + 1):
                    try:
                        jobs = fetch_jobs_page(
                            client=client,
                            app_id=app_id,
                            app_key=app_key,
                            search_term=search_term,
                            location_filter=location_filter,
                            page=page,
                            results_per_page=results_per_page,
                        )
                    except httpx.HTTPStatusError as exc:
                        print(
                            f"HTTP error for term='{search_term}' search_location='{search_location}' "
                            f"location_filter_applied={location_filter_applied} page={page}: "
                            f"{exc.response.status_code} {exc.response.text}",
                            file=sys.stderr,
                        )
                        continue
                    except httpx.RequestError as exc:
                        print(
                            f"Request error for term='{search_term}' search_location='{search_location}' "
                            f"location_filter_applied={location_filter_applied} page={page}: {exc}",
                            file=sys.stderr,
                        )
                        continue
                    except ValueError as exc:
                        print(
                            f"Response parsing error for term='{search_term}' "
                            f"search_location='{search_location}' "
                            f"location_filter_applied={location_filter_applied} page={page}: {exc}",
                            file=sys.stderr,
                        )
                        continue

                    combo_rows_fetched += len(jobs)
                    records.extend(
                        normalize_job(
                            job=job,
                            search_term=search_term,
                            search_location=search_location,
                            location_filter_applied=location_filter_applied,
                            ingested_at=ingested_at,
                        )
                        for job in jobs
                    )

                    # A short sleep is enough for this weekend project and avoids hammering the API.
                    time.sleep(sleep_seconds)

                print(
                    f"Fetched term='{search_term}' search_location='{search_location}' "
                    f"location_filter_applied={location_filter_applied} rows={combo_rows_fetched}"
                )

    return build_jobs_dataframe(records)


def main() -> None:
    """Run the ingestion step and save the output locally."""
    args = parse_args()

    try:
        jobs_df = ingest_jobs(
            pages=args.pages,
            results_per_page=args.results_per_page,
            sleep_seconds=args.sleep_seconds,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    output_path = get_output_path()
    fetched_count = len(jobs_df)

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        combined_df = pd.concat([existing_df, jobs_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["job_id"], keep="last")
        new_jobs_added = len(combined_df) - len(existing_df)
    else:
        combined_df = jobs_df.drop_duplicates(subset=["job_id"], keep="last")
        new_jobs_added = len(combined_df)

    combined_df.to_csv(output_path, index=False)

    print(f"Jobs fetched: {fetched_count}")
    print(f"New jobs added: {new_jobs_added}")
    print(f"Final deduplicated row count: {len(combined_df)}")
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
