"""Retrieve top company contexts from the lightweight RAG index."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_index_path() -> Path:
    return get_repo_root() / "data" / "processed" / "company_rag_index.pkl"


@st.cache_resource(show_spinner=False)
def load_index() -> dict:
    index_path = get_index_path()
    if not index_path.exists():
        raise FileNotFoundError(f"RAG index not found: {index_path}")
    with index_path.open("rb") as file:
        return pickle.load(file)


@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def safe_float(value: object) -> float:
    try:
        if value is None:
            return 0.0
        if isinstance(value, float) and np.isnan(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_values(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [0.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def is_valid_company_document(document: dict) -> bool:
    """Filter out companies dominated by training_or_spam if category mix is available."""
    category_mix = document.get("job_type_category_mix") or {}
    training_count = safe_float(category_mix.get("training_or_spam", 0))
    employer_count = safe_float(category_mix.get("employer", 0))
    recruiter_count = safe_float(category_mix.get("recruiter", 0))
    total_count = training_count + employer_count + recruiter_count

    if total_count == 0:
        return True

    return training_count < total_count


def retrieve_top_contexts(query: str, top_k: int = 5) -> list[dict]:
    """Embed a user query, rerank with simple business metrics, and return top company documents."""
    index_data = load_index()
    model = load_model(index_data["model_name"])
    query_embedding = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]

    document_embeddings = np.asarray(index_data["embeddings"])
    semantic_scores = document_embeddings @ query_embedding

    candidate_rows = []
    for index, document in enumerate(index_data["documents"]):
        if not is_valid_company_document(document):
            continue
        candidate_rows.append(
            {
                "index": index,
                "document": document,
                "semantic_score": float(semantic_scores[index]),
                "posting_count": safe_float(document.get("posting_count")),
                "avg_salary_gap_pct": safe_float(document.get("avg_salary_gap_pct")),
                "external_rating": safe_float(document.get("external_rating")),
            }
        )

    posting_values = [row["posting_count"] for row in candidate_rows]
    gap_values = [row["avg_salary_gap_pct"] for row in candidate_rows]
    rating_values = [row["external_rating"] for row in candidate_rows]

    normalized_posting = normalize_values(posting_values)
    normalized_gap = normalize_values(gap_values)
    normalized_rating = normalize_values(rating_values)

    for row, posting_score, gap_score, rating_score in zip(
        candidate_rows,
        normalized_posting,
        normalized_gap,
        normalized_rating,
    ):
        row["final_score"] = (
            0.65 * row["semantic_score"]
            + 0.15 * posting_score
            + 0.10 * gap_score
            + 0.10 * rating_score
        )

    candidate_rows.sort(key=lambda row: row["final_score"], reverse=True)
    results = []
    for row in candidate_rows[:top_k]:
        document = row["document"]
        results.append(
            {
                "company_lower": document["company_lower"],
                "document_text": document["document_text"],
                "semantic_score": row["semantic_score"],
                "final_score": row["final_score"],
                "source_fields": {
                    "posting_count": document.get("posting_count"),
                    "avg_salary_gap_pct": document.get("avg_salary_gap_pct"),
                    "external_rating": document.get("external_rating"),
                    "external_review_count": document.get("external_review_count"),
                    "top_job_titles": document.get("top_job_titles"),
                    "seniority_mix": document.get("seniority_mix"),
                    "role_mix": document.get("role_mix"),
                    "location_mix": document.get("location_mix"),
                    "salary_min": document.get("salary_min"),
                    "salary_median": document.get("salary_median"),
                    "salary_max": document.get("salary_max"),
                    "sample_above_market_roles": document.get("sample_above_market_roles"),
                    "sample_below_market_roles": document.get("sample_below_market_roles"),
                },
            }
        )
    return results
