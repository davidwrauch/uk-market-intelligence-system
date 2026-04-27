"""Quick validation summary for the local raw jobs CSV."""

from pathlib import Path

import pandas as pd


REMOTE_PATTERNS = [
    "remote",
    "work from home",
    "wfh",
    "home based",
    "anywhere in the uk",
    "uk remote",
]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    raw_jobs_path = get_repo_root() / "data" / "raw" / "raw_jobs.csv"
    df = pd.read_csv(raw_jobs_path)
    duplicate_job_ids = len(df) - df["job_id"].nunique()

    print("Raw Jobs Validation Summary")
    print("===========================")
    print(f"Total rows: {len(df):,}")
    print(f"Deduplicated job_id count: {df['job_id'].nunique():,}")
    print(f"Duplicated job_ids before deduplication: {duplicate_job_ids:,}")
    print()

    if "search_location" in df.columns:
        print("Count by search location")
        print(df["search_location"].fillna("unknown").value_counts().to_string())
        print()

    if "location_filter_applied" in df.columns:
        print("Count by location_filter_applied")
        print(df["location_filter_applied"].fillna("unknown").value_counts().to_string())
        print()

    if "search_term" in df.columns:
        print("Count by search term")
        print(df["search_term"].fillna("unknown").value_counts().to_string())
        print()

    if "search_location" in df.columns and "search_term" in df.columns:
        print("Count by search location and search term")
        combo_counts = (
            df.groupby(["search_location", "search_term"], dropna=False)
            .size()
            .sort_values(ascending=False)
        )
        print(combo_counts.to_string())
        print()

    print("Top 25 companies by job count")
    print(df["company"].fillna("unknown").value_counts().head(25).to_string())
    print()

    has_salary = df["salary_min"].notna() | df["salary_max"].notna()
    print(f"Jobs with salary_min or salary_max present: {int(has_salary.sum()):,}")

    text_columns = []
    for column in ["title", "description", "location", "search_location"]:
        if column in df.columns:
            text_columns.append(df[column].fillna("").astype(str))

    combined_text = pd.Series("", index=df.index)
    for series in text_columns:
        combined_text = combined_text + " " + series.str.lower()

    remote_mask = pd.Series(False, index=df.index)
    for pattern in REMOTE_PATTERNS:
        remote_mask = remote_mask | combined_text.str.contains(pattern, regex=False)

    print(f"Jobs that appear remote: {int(remote_mask.sum()):,}")
    print()

    print("Count by location field (top 25)")
    print(df["location"].fillna("unknown").value_counts().head(25).to_string())


if __name__ == "__main__":
    main()
