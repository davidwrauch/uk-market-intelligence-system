"""Build a lightweight file-based RAG index for company-level market context."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_paths() -> tuple[Path, Path, Path]:
    processed_dir = get_repo_root() / "data" / "processed"
    return (
        processed_dir / "company_signals.csv",
        processed_dir / "scored_jobs.csv",
        processed_dir / "company_rag_index.pkl",
    )


def format_percentage(value: object) -> str:
    if pd.isna(value):
        return "not available"
    return f"{float(value) * 100:.1f}%"


def format_number(value: object) -> str:
    if pd.isna(value):
        return "not available"
    return f"{float(value):,.0f}"


def classify_seniority(title: str) -> str:
    lowered = title.lower()
    if any(token in lowered for token in ["principal", "head", "director"]):
        return "principal/head/director"
    if any(token in lowered for token in ["senior", "lead", "manager"]):
        return "senior/lead/manager"
    if any(token in lowered for token in ["junior", "graduate", "entry"]):
        return "junior/graduate/entry"
    return "mid-level/other"


def classify_role(title: str) -> str:
    lowered = title.lower()
    if "data scientist" in lowered:
        return "data scientist"
    if "machine learning" in lowered or " ml " in f" {lowered} ":
        return "machine learning"
    if " ai " in f" {lowered} " or lowered.startswith("ai ") or "artificial intelligence" in lowered:
        return "AI"
    if "business intelligence" in lowered or " bi " in f" {lowered} ":
        return "BI"
    if "analytics" in lowered:
        return "analytics"
    if "analyst" in lowered:
        return "analyst"
    if "engineer" in lowered:
        return "engineer"
    return "other"


def summarize_counts(series: pd.Series) -> dict:
    if series.empty:
        return {}
    return series.value_counts().to_dict()


def build_company_document(company_row: pd.Series, company_jobs_df: pd.DataFrame) -> dict:
    """Create one text document per company from processed company and job outputs."""
    company_name = str(company_row.get("company_lower") or "unknown")
    titles_series = company_jobs_df["title"].dropna().astype(str) if "title" in company_jobs_df.columns else pd.Series(dtype=str)
    top_job_titles = (
        titles_series.value_counts().head(5).index.tolist() if not titles_series.empty else []
    )
    search_locations = (
        sorted(company_jobs_df["search_location"].dropna().astype(str).unique().tolist())
        if "search_location" in company_jobs_df.columns
        else []
    )
    category_mix = (
        company_jobs_df["job_type_category"].fillna("unknown").value_counts().to_dict()
        if "job_type_category" in company_jobs_df.columns
        else {}
    )
    snippets = []
    if "description" in company_jobs_df.columns:
        snippets = company_jobs_df["description"].dropna().astype(str).head(3).tolist()

    seniority_mix = summarize_counts(titles_series.map(classify_seniority)) if not titles_series.empty else {}
    role_mix = summarize_counts(titles_series.map(classify_role)) if not titles_series.empty else {}
    location_mix = summarize_counts(company_jobs_df["search_location"].fillna("unknown").astype(str)) if "search_location" in company_jobs_df.columns else {}

    salary_series = pd.to_numeric(company_jobs_df.get("salary"), errors="coerce") if "salary" in company_jobs_df.columns else pd.Series(dtype=float)
    salary_min = salary_series.min() if not salary_series.empty else None
    salary_median = salary_series.median() if not salary_series.empty else None
    salary_max = salary_series.max() if not salary_series.empty else None

    above_market_roles = []
    below_market_roles = []
    if {"title", "salary_gap_pct"}.issubset(company_jobs_df.columns):
        ranked_df = company_jobs_df[["title", "salary_gap_pct"]].copy()
        ranked_df["salary_gap_pct"] = pd.to_numeric(ranked_df["salary_gap_pct"], errors="coerce")
        ranked_df = ranked_df.dropna(subset=["title", "salary_gap_pct"])
        above_market_roles = (
            ranked_df.sort_values("salary_gap_pct", ascending=True)["title"].head(3).astype(str).tolist()
        )
        below_market_roles = (
            ranked_df.sort_values("salary_gap_pct", ascending=False)["title"].head(3).astype(str).tolist()
        )

    document_text = "\n".join(
        [
            f"Company: {company_name}",
            f"Posting count: {company_row.get('posting_count', 'not available')}",
            f"Average salary gap: {format_percentage(company_row.get('avg_salary_gap_pct'))}",
            f"External rating: {company_row.get('external_rating', 'not available')}",
            f"External review count: {format_number(company_row.get('external_review_count'))}",
            f"Top job titles: {', '.join(top_job_titles) if top_job_titles else 'not available'}",
            f"Search locations: {', '.join(search_locations) if search_locations else 'not available'}",
            f"Seniority mix: {seniority_mix if seniority_mix else 'not available'}",
            f"Role mix: {role_mix if role_mix else 'not available'}",
            f"Location mix: {location_mix if location_mix else 'not available'}",
            f"Job type category mix: {category_mix if category_mix else 'not available'}",
            f"Salary range: min {format_number(salary_min)}, median {format_number(salary_median)}, max {format_number(salary_max)}",
            f"Sample above-market roles: {', '.join(above_market_roles) if above_market_roles else 'not available'}",
            f"Sample below-market roles: {', '.join(below_market_roles) if below_market_roles else 'not available'}",
            f"Description snippets: {' | '.join(snippets) if snippets else 'not available in processed data'}",
        ]
    )

    return {
        "company_lower": company_name,
        "document_text": document_text,
        "posting_count": company_row.get("posting_count"),
        "avg_salary_gap_pct": company_row.get("avg_salary_gap_pct"),
        "external_rating": company_row.get("external_rating"),
        "external_review_count": company_row.get("external_review_count"),
        "top_job_titles": top_job_titles,
        "search_locations": search_locations,
        "seniority_mix": seniority_mix,
        "role_mix": role_mix,
        "location_mix": location_mix,
        "job_type_category_mix": category_mix,
        "salary_min": salary_min,
        "salary_median": salary_median,
        "salary_max": salary_max,
        "sample_above_market_roles": above_market_roles,
        "sample_below_market_roles": below_market_roles,
    }


def build_index() -> dict:
    company_path, jobs_path, index_path = get_paths()
    company_df = pd.read_csv(company_path)
    jobs_df = pd.read_csv(jobs_path)

    documents = []
    for _, company_row in company_df.iterrows():
        company_name = company_row.get("company_lower")
        company_jobs_df = jobs_df[jobs_df["company_lower"] == company_name].copy()
        documents.append(build_company_document(company_row, company_jobs_df))

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [document["document_text"] for document in documents],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    index_data = {
        "model_name": MODEL_NAME,
        "documents": documents,
        "embeddings": embeddings,
    }

    with index_path.open("wb") as file:
        pickle.dump(index_data, file)

    return index_data


def main() -> None:
    index_data = build_index()
    print(f"Number of company documents indexed: {len(index_data['documents'])}")

    from src.rag.retrieve_context import retrieve_top_contexts

    sample_results = retrieve_top_contexts(
        query="Which companies look strongest for data science roles?",
        top_k=5,
    )
    sample_companies = [result["company_lower"] for result in sample_results]
    print("Sample retrieved companies for query: Which companies look strongest for data science roles?")
    print(", ".join(sample_companies) if sample_companies else "No companies retrieved.")


if __name__ == "__main__":
    main()
