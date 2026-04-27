"""Quick validation summary for scored job outputs."""

from pathlib import Path

import pandas as pd


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    processed_dir = get_repo_root() / "data" / "processed"
    scored_jobs_path = processed_dir / "scored_jobs.csv"

    df = pd.read_csv(scored_jobs_path)

    print("Scored Jobs Validation Summary")
    print("==============================")

    print("Counts by job_type_category")
    print(df["job_type_category"].fillna("unknown").value_counts().to_string())
    print()

    print("Counts by is_valid_job")
    print(df["is_valid_job"].fillna("unknown").value_counts().to_string())
    print()

    print("Top 25 companies overall")
    print(df["company_lower"].fillna("unknown").value_counts().head(25).to_string())
    print()

    valid_df = df[df["is_valid_job"].fillna(False).astype(bool)].copy()
    print("Top 25 companies where is_valid_job = True")
    print(valid_df["company_lower"].fillna("unknown").value_counts().head(25).to_string())
    print()

    print("Average salary and predicted salary by job_type_category")
    salary_summary = (
        df.groupby("job_type_category", dropna=False)[["salary", "predicted_salary"]]
        .mean()
        .round(2)
        .sort_index()
    )
    print(salary_summary.to_string())
    print()

    print("Average salary_gap by job_type_category")
    gap_summary = (
        df.groupby("job_type_category", dropna=False)["salary_gap"]
        .mean()
        .round(2)
        .sort_index()
    )
    print(gap_summary.to_string())
    print()

    print("Count by search_location and job_type_category")
    location_category_summary = (
        df.groupby(["search_location", "job_type_category"], dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    print(location_category_summary.to_string())


if __name__ == "__main__":
    main()
