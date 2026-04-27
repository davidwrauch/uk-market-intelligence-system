"""Create lightweight external rating signals from company-level data."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv


GOOGLE_PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"


def get_repo_root() -> Path:
    """Resolve the repository root from this file's location."""
    return Path(__file__).resolve().parents[2]


def get_paths() -> tuple[Path, Path]:
    """Return input and output paths for company trust enrichment."""
    processed_dir = get_repo_root() / "data" / "processed"
    return (
        processed_dir / "company_signals.csv",
        processed_dir / "company_trust_signals.csv",
    )


def get_scored_jobs_path() -> Path:
    """Return the scored jobs path used to recover company job categories when needed."""
    return get_repo_root() / "data" / "processed" / "scored_jobs.csv"


def fetch_google_places_metrics(
    client: httpx.Client,
    api_key: str,
    company_name: str,
) -> tuple[float | None, int | None, str]:
    """Fetch Google Places rating metrics for a company name."""
    response = client.post(
        GOOGLE_PLACES_SEARCH_URL,
        headers={
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.displayName,places.rating,places.userRatingCount",
            "Content-Type": "application/json",
        },
        json={"textQuery": company_name},
    )
    response.raise_for_status()

    payload = response.json()
    places = payload.get("places", []) if isinstance(payload, dict) else []
    if not places:
        return None, None, "no_match"

    place = places[0]
    external_rating = place.get("rating")
    external_review_count = place.get("userRatingCount")

    if external_rating is None and external_review_count is None:
        return None, None, "no_match"

    return external_rating, external_review_count, "api"


def attach_company_job_type_category(company_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure company-level input has a job_type_category column.

    If company_signals does not already include it, derive a simple company-level category
    from scored_jobs with priority: training_or_spam > recruiter > employer.
    """
    if "job_type_category" in company_df.columns:
        return company_df

    scored_jobs_path = get_scored_jobs_path()
    if not scored_jobs_path.exists():
        company_df["job_type_category"] = pd.NA
        return company_df

    scored_jobs_df = pd.read_csv(scored_jobs_path, usecols=["company_lower", "job_type_category"])
    priority = {"training_or_spam": 3, "recruiter": 2, "employer": 1}
    scored_jobs_df["category_priority"] = scored_jobs_df["job_type_category"].map(priority).fillna(0)
    company_categories = (
        scored_jobs_df.sort_values("category_priority", ascending=False)
        .drop_duplicates(subset=["company_lower"], keep="first")
        [["company_lower", "job_type_category"]]
    )
    return company_df.merge(company_categories, on="company_lower", how="left")


def build_trust_signals(company_df: pd.DataFrame) -> pd.DataFrame:
    """Create a simple company-level external rating output."""
    load_dotenv()
    google_places_api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    company_df = attach_company_job_type_category(company_df.copy())

    top_companies = company_df.sort_values("posting_count", ascending=False).head(50).copy()
    top_companies["external_rating"] = pd.NA
    top_companies["external_review_count"] = pd.NA
    top_companies["external_rating_source"] = "google_places"

    if not google_places_api_key:
        top_companies["external_match_method"] = "not_configured"
    else:
        top_companies["external_match_method"] = "no_match"

        with httpx.Client(timeout=20.0) as client:
            for index, row in top_companies.iterrows():
                company_name = str(row.get("company_lower") or "").strip()
                if not company_name:
                    top_companies.at[index, "external_match_method"] = "no_match"
                    continue

                try:
                    external_rating, review_count, match_method = fetch_google_places_metrics(
                        client=client,
                        api_key=google_places_api_key,
                        company_name=company_name,
                    )
                except httpx.HTTPError:
                    external_rating, review_count, match_method = None, None, "no_match"

                top_companies.at[index, "external_rating"] = external_rating
                top_companies.at[index, "external_review_count"] = review_count
                top_companies.at[index, "external_match_method"] = match_method

    filtered_mask = top_companies["job_type_category"].eq("training_or_spam")
    top_companies.loc[filtered_mask, "external_rating"] = pd.NA
    top_companies.loc[filtered_mask, "external_review_count"] = pd.NA
    top_companies.loc[filtered_mask, "external_match_method"] = "filtered"

    return top_companies[
        [
            "company_lower",
            "job_type_category",
            "posting_count",
            "avg_salary_gap_pct",
            "external_rating",
            "external_review_count",
            "external_rating_source",
            "external_match_method",
        ]
    ]


def run_enrichment() -> pd.DataFrame:
    """Run the lightweight company external rating enrichment and save the output."""
    input_path, output_path = get_paths()

    if not input_path.exists():
        raise FileNotFoundError(f"Company signals file not found: {input_path}")

    company_df = pd.read_csv(input_path)
    enrichment_df = build_trust_signals(company_df)
    enrichment_df.to_csv(output_path, index=False)
    return enrichment_df


def main() -> None:
    """Run the lightweight company external rating enrichment step."""
    _, output_path = get_paths()
    enrichment_df = run_enrichment()

    print(f"Companies enriched: {len(enrichment_df)}")
    print(f"External rating signals saved to {output_path}")


if __name__ == "__main__":
    main()
