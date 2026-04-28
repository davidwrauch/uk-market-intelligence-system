"""Train a simple salary model from BigQuery feature data and save scored outputs."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.enrichment.enrich_trust_signals import run_enrichment
from src.pipelines.bigquery_utils import get_bigquery_client


NUMERIC_FEATURES = [
    "senior_flag",
    "lead_flag",
    "ml_flag",
    "analyst_flag",
]

CATEGORICAL_FEATURES = [
    "company_lower",
    "location",
]


def get_repo_root() -> Path:
    """Resolve the repository root from this file's location."""
    return Path(__file__).resolve().parents[2]


def get_output_paths() -> tuple[Path, Path]:
    """Return output CSV paths and ensure the processed directory exists."""
    processed_dir = get_repo_root() / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir / "scored_jobs.csv", processed_dir / "company_signals.csv"


def get_model_path() -> Path:
    """Return the saved model path and ensure the models directory exists."""
    models_dir = get_repo_root() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir / "salary_model.joblib"


def read_feature_jobs() -> pd.DataFrame:
    """Read the feature_jobs table from BigQuery into pandas."""
    load_dotenv()
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset = os.getenv("BIGQUERY_DATASET")

    if not project_id:
        raise ValueError("Missing GCP_PROJECT_ID environment variable.")
    if not dataset:
        raise ValueError("Missing BIGQUERY_DATASET environment variable.")

    client = get_bigquery_client()
    query = f"SELECT * FROM `{project_id}.{dataset}.feature_jobs`"
    return client.query(query).to_dataframe()


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a usable salary target and normalize simple feature types."""
    working_df = df.copy()

    if "job_type_category" in working_df.columns:
        category_counts = working_df["job_type_category"].fillna("unknown").value_counts()
        print("Counts by job_type_category:")
        print(category_counts.to_string())

    if "is_valid_job" in working_df.columns:
        invalid_rows = (~working_df["is_valid_job"].fillna(True)).sum()
        print(f"Rows flagged as training_or_spam: {int(invalid_rows)}")
        print("No rows are being removed yet; this is an inspection-only filter step.")

    working_df["salary"] = pd.to_numeric(working_df["salary"], errors="coerce")
    working_df = working_df[working_df["salary"].between(20000, 200000)].copy()
    scoring_df = working_df.copy()

    if "is_valid_job" in working_df.columns:
        working_df = working_df[working_df["is_valid_job"].fillna(True)].copy()

    for column in NUMERIC_FEATURES:
        working_df[column] = working_df[column].fillna(False).astype(int)
        scoring_df[column] = scoring_df[column].fillna(False).astype(int)

    for column in CATEGORICAL_FEATURES:
        working_df[column] = working_df[column].fillna("unknown").astype(str)
        scoring_df[column] = scoring_df[column].fillna("unknown").astype(str)

    return working_df, scoring_df


def build_model() -> Pipeline:
    """Create a simple regression pipeline with a small amount of preprocessing."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="most_frequent"), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    """Train the model and print a simple MAE evaluation."""
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_columns]
    y = df["salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Test MAE: {mae:,.2f}")

    return model


def score_jobs(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Generate predicted salary and simple salary gap metrics for each row."""
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    scored_df = df.copy()
    scored_df["predicted_salary"] = model.predict(scored_df[feature_columns])
    scored_df["salary_gap"] = scored_df["salary"] - scored_df["predicted_salary"]
    scored_df["salary_gap_pct"] = scored_df["salary_gap"] / scored_df["predicted_salary"]
    scored_df["salary_gap_pct"] = scored_df["salary_gap_pct"].clip(-1, 1)
    scored_df["abs_salary_gap_pct"] = scored_df["salary_gap_pct"].abs()

    return scored_df[
        [
            "job_id",
            "title",
            "company_lower",
            "location",
            "search_location",
            "redirect_url",
            "description",
            "job_type_category",
            "is_valid_job",
            "salary",
            "predicted_salary",
            "salary_gap",
            "salary_gap_pct",
            "abs_salary_gap_pct",
            "senior_flag",
            "lead_flag",
            "ml_flag",
            "analyst_flag",
        ]
    ]


def build_company_signals(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate job-level salary gaps into a simple company signal table."""
    company_df = (
        scored_df.groupby("company_lower", dropna=False)
        .agg(
            posting_count=("job_id", "count"),
            avg_salary_gap_pct=("salary_gap_pct", "mean"),
            avg_abs_salary_gap_pct=("abs_salary_gap_pct", "mean"),
            median_salary=("salary", "median"),
            median_predicted_salary=("predicted_salary", "median"),
        )
        .reset_index()
        .sort_values(["posting_count", "avg_salary_gap_pct"], ascending=[False, False])
    )
    return company_df


def main() -> None:
    """Run the simple model training and scoring workflow."""
    feature_jobs_df = read_feature_jobs()
    training_df, scoring_df = prepare_training_data(feature_jobs_df)

    if training_df.empty:
        print("No training data available, skipping model step.")
return

    print(f"Training rows count: {len(training_df)}")
    model = train_and_evaluate(training_df)
    scored_jobs_df = score_jobs(scoring_df, model)
    print(f"Total rows scored: {len(scored_jobs_df)}")
    company_signals_df = build_company_signals(scored_jobs_df)

    scored_jobs_path, company_signals_path = get_output_paths()
    model_path = get_model_path()

    scored_jobs_df.to_csv(scored_jobs_path, index=False)
    company_signals_df.to_csv(company_signals_path, index=False)
    enrichment_df = run_enrichment()
    company_signals_with_external_df = company_signals_df.merge(
        enrichment_df[
            [
                "company_lower",
                "external_rating",
                "external_review_count",
                "external_rating_source",
                "external_match_method",
            ]
        ],
        on="company_lower",
        how="left",
    )
    company_signals_with_external_df.to_csv(company_signals_path, index=False)
    joblib.dump(model, model_path)

    print(f"Scored jobs saved to {scored_jobs_path}")
    print(f"Company signals saved to {company_signals_path}")
    print(f"Company external rating signals saved with {len(enrichment_df)} rows")
    print(f"Model saved to {model_path}")
    print("Training and scoring completed successfully.")


if __name__ == "__main__":
    main()
