import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.rag.retrieve_context import retrieve_top_contexts


load_dotenv()
ANTHROPIC_MODEL = "claude-haiku-4-5"

st.set_page_config(page_title="UK Market Intelligence System", layout="wide")


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_processed_path(filename: str) -> Path:
    return get_repo_root() / "data" / "processed" / filename


@st.cache_data(show_spinner=False)
def load_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            st.error(f"Missing file: {path}")
            st.stop()
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_last_updated_text() -> str:
    processed_dir = get_processed_path("")
    candidate_files = [
        get_processed_path("scored_jobs.csv"),
        get_processed_path("company_signals.csv"),
    ]

    latest_timestamp = None
    for path in candidate_files:
        if path.exists():
            modified_timestamp = pd.Timestamp(path.stat().st_mtime, unit="s")
            if latest_timestamp is None or modified_timestamp > latest_timestamp:
                latest_timestamp = modified_timestamp

    if latest_timestamp is None:
        return "Last updated: unavailable"

    return f"Last updated: {latest_timestamp.strftime('%Y-%m-%d %H:%M')}"


def format_company_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("-", " ").strip().title()


def format_gbp(value: object) -> str:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return "No salary data"
    return f"GBP {float(numeric_value):,.0f}"


def format_percent_abs(value: object) -> str:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return "No pay data"
    return f"{abs(float(numeric_value)) * 100:.1f}%"


def prepare_jobs_dataframe(jobs_df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = jobs_df.copy()
    prepared_df["salary_gap_pct"] = pd.to_numeric(prepared_df.get("salary_gap_pct"), errors="coerce")
    prepared_df["salary"] = pd.to_numeric(prepared_df.get("salary"), errors="coerce")
    prepared_df["predicted_salary"] = pd.to_numeric(
        prepared_df.get("predicted_salary"), errors="coerce"
    )

    if "is_valid_job" in prepared_df.columns:
        prepared_df = prepared_df[prepared_df["is_valid_job"].fillna(False).astype(bool)].copy()

    if "search_location" in prepared_df.columns:
        prepared_df["search_location"] = prepared_df["search_location"].fillna("unknown")
    else:
        prepared_df["search_location"] = "unknown"

    if "location" in prepared_df.columns:
        prepared_df["location"] = prepared_df["location"].fillna("unknown")
    else:
        prepared_df["location"] = "unknown"

    if "title" in prepared_df.columns:
        prepared_df["title"] = prepared_df["title"].fillna("")
    else:
        prepared_df["title"] = ""

    if "company_lower" in prepared_df.columns:
        prepared_df["company_lower"] = prepared_df["company_lower"].fillna("")
    else:
        prepared_df["company_lower"] = ""

    if "redirect_url" in prepared_df.columns:
        prepared_df["redirect_url"] = prepared_df["redirect_url"].fillna("")
    elif "adzuna_url" in prepared_df.columns:
        prepared_df["redirect_url"] = prepared_df["adzuna_url"].fillna("")
    elif "url" in prepared_df.columns:
        prepared_df["redirect_url"] = prepared_df["url"].fillna("")
    elif "adref" in prepared_df.columns:
        prepared_df["redirect_url"] = prepared_df["adref"].fillna("")
    else:
        prepared_df["redirect_url"] = ""

    if "description" in prepared_df.columns:
        prepared_df["description"] = prepared_df["description"].fillna("")
    else:
        prepared_df["description"] = ""

    return prepared_df


def merge_company_data(company_df: pd.DataFrame, enrichment_df: pd.DataFrame) -> pd.DataFrame:
    if enrichment_df.empty or "company_lower" not in enrichment_df.columns:
        return company_df.copy()

    merged_df = company_df.copy()
    replaceable_columns = [
        column
        for column in [
            "external_rating",
            "external_review_count",
            "external_rating_source",
            "external_match_method",
        ]
        if column in merged_df.columns
    ]
    if replaceable_columns:
        merged_df = merged_df.drop(columns=replaceable_columns)

    enrichment_columns = [
        column
        for column in [
            "company_lower",
            "external_rating",
            "external_review_count",
            "external_rating_source",
            "external_match_method",
        ]
        if column in enrichment_df.columns
    ]
    return merged_df.merge(enrichment_df[enrichment_columns], on="company_lower", how="left")


def prepare_company_dataframe(company_df: pd.DataFrame, jobs_df: pd.DataFrame) -> pd.DataFrame:
    valid_companies = jobs_df["company_lower"].dropna().astype(str).unique().tolist()
    prepared_df = company_df[company_df["company_lower"].isin(valid_companies)].copy()
    prepared_df["posting_count"] = pd.to_numeric(prepared_df.get("posting_count"), errors="coerce").fillna(0)
    prepared_df["avg_salary_gap_pct"] = pd.to_numeric(
        prepared_df.get("avg_salary_gap_pct"), errors="coerce"
    )
    prepared_df["median_salary"] = pd.to_numeric(prepared_df.get("median_salary"), errors="coerce")
    prepared_df["median_predicted_salary"] = pd.to_numeric(
        prepared_df.get("median_predicted_salary"), errors="coerce"
    )
    prepared_df["external_rating"] = pd.to_numeric(
        prepared_df.get("external_rating"), errors="coerce"
    )
    prepared_df["external_review_count"] = pd.to_numeric(
        prepared_df.get("external_review_count"), errors="coerce"
    )
    prepared_df["company_score"] = prepared_df.apply(score_company_row, axis=1)
    prepared_df = prepared_df.sort_values(
        ["company_score", "posting_count"],
        ascending=[False, False],
    )
    return prepared_df


def get_hiring_level(posting_count: object) -> tuple[str, int]:
    count = int(pd.to_numeric(posting_count, errors="coerce") or 0)
    if count >= 100:
        return f"High ({count} jobs)", 2
    if count >= 30:
        return f"Medium ({count} jobs)", 1
    return f"Low ({count} jobs)", 0


def get_pay_vs_market_label(avg_salary_gap_pct: object) -> tuple[str, int]:
    gap_value = pd.to_numeric(avg_salary_gap_pct, errors="coerce")
    if pd.isna(gap_value):
        return "No pay data", 0

    if float(gap_value) < -0.02:
        return f"🟢 {abs(float(gap_value)) * 100:.1f}% above market", 2
    if float(gap_value) > 0.02:
        return f"🔴 {abs(float(gap_value)) * 100:.1f}% below market", 0
    return "🟡 about market", 1


def get_reviews_label(external_rating: object, external_review_count: object) -> tuple[str, float]:
    rating_value = pd.to_numeric(external_rating, errors="coerce")
    review_count = pd.to_numeric(external_review_count, errors="coerce")

    if pd.isna(rating_value) or pd.isna(review_count):
        return "No review data", 0.0

    count_int = int(review_count)
    if count_int < 10:
        return f"\u2b50 {float(rating_value):.1f} ({count_int} reviews - low data)", float(rating_value)
    return f"\u2b50 {float(rating_value):.1f} ({count_int} reviews)", float(rating_value)


def score_company_row(row: pd.Series) -> float:
    _, hiring_score = get_hiring_level(row.get("posting_count"))
    _, pay_score = get_pay_vs_market_label(row.get("avg_salary_gap_pct"))
    _, rating_score = get_reviews_label(row.get("external_rating"), row.get("external_review_count"))
    return float(hiring_score) + float(pay_score) + (rating_score / 5.0 if rating_score else 0.0)


def build_company_display_table(company_df: pd.DataFrame) -> pd.DataFrame:
    display_df = pd.DataFrame()
    display_df["Company"] = company_df["company_lower"].map(format_company_name)
    display_df["Hiring"] = company_df["posting_count"].map(lambda value: get_hiring_level(value)[0])
    display_df["Pay vs Market"] = company_df["avg_salary_gap_pct"].map(
        lambda value: get_pay_vs_market_label(value)[0]
    )
    display_df["Estimated Market Pay"] = company_df["median_predicted_salary"].map(format_gbp)
    display_df["Listed Avg Pay"] = company_df["median_salary"].map(format_gbp)
    display_df["Reviews"] = company_df.apply(
        lambda row: get_reviews_label(row.get("external_rating"), row.get("external_review_count"))[0],
        axis=1,
    )
    return display_df


def build_jobs_display_table(jobs_df: pd.DataFrame) -> pd.DataFrame:
    display_df = pd.DataFrame()
    display_df["Job Title"] = jobs_df["title"].fillna("")
    display_df["Company"] = jobs_df["company_lower"].map(format_company_name)
    display_df["Listed Salary"] = jobs_df["salary"].map(format_gbp)
    display_df["Estimated Market Salary"] = jobs_df["predicted_salary"].map(format_gbp)
    display_df["Pay vs Market"] = jobs_df["salary_gap_pct"].map(
        lambda value: get_pay_vs_market_label(value)[0]
    )
    display_df["External Link"] = jobs_df["redirect_url"].map(
        lambda value: value if pd.notna(value) and str(value).strip() else "No link"
    )
    display_df["Location"] = jobs_df["location"].fillna("unknown")
    display_df["Search Area"] = jobs_df["search_location"].fillna("unknown")
    return display_df


def render_top_summary(company_df: pd.DataFrame) -> None:
    if company_df.empty:
        return

    best_row = company_df.iloc[0]
    pay_label, _ = get_pay_vs_market_label(best_row.get("avg_salary_gap_pct"))
    hiring_label, _ = get_hiring_level(best_row.get("posting_count"))
    reviews_label, _ = get_reviews_label(
        best_row.get("external_rating"),
        best_row.get("external_review_count"),
    )

    with st.container(border=True):
        st.markdown("**Best Overall Company**")
        st.markdown(f"### {format_company_name(best_row.get('company_lower'))}")
        st.write(f"Pay: {pay_label}")
        st.write(f"Reviews: {reviews_label}")
        st.write(f"Hiring: {hiring_label}")


def summarize_company_evidence(company_jobs_df: pd.DataFrame) -> dict[str, Any]:
    title_series = company_jobs_df["title"].dropna().astype(str) if "title" in company_jobs_df.columns else pd.Series(dtype=str)
    lower_titles = title_series.str.lower()

    def count_matches(tokens: list[str]) -> int:
        return int(lower_titles.str.contains("|".join(tokens), regex=True).sum()) if not lower_titles.empty else 0

    seniority_mix = {
        "senior/lead/principal/head/director/manager": count_matches(
            ["senior", "lead", "principal", "head", "director", "manager"]
        ),
        "junior/graduate/entry": count_matches(["junior", "graduate", "entry"]),
    }
    role_mix = {
        "data scientist": count_matches(["data scientist"]),
        "machine learning": count_matches(["machine learning", r"\bml\b"]),
        "AI": count_matches([r"\bai\b", "artificial intelligence"]),
        "analytics": count_matches(["analytics"]),
        "BI": count_matches(["business intelligence", r"\bbi\b"]),
        "analyst": count_matches(["analyst"]),
        "engineer": count_matches(["engineer"]),
    }
    location_mix = (
        company_jobs_df["search_location"].fillna("unknown").astype(str).value_counts().to_dict()
        if "search_location" in company_jobs_df.columns
        else {}
    )
    salary_series = pd.to_numeric(company_jobs_df.get("salary"), errors="coerce")
    salary_min = salary_series.min() if not salary_series.empty else None
    salary_median = salary_series.median() if not salary_series.empty else None
    salary_max = salary_series.max() if not salary_series.empty else None

    above_market_roles = []
    below_market_roles = []
    if {"title", "salary_gap_pct"}.issubset(company_jobs_df.columns):
        ranked_df = company_jobs_df[["title", "salary_gap_pct"]].copy()
        ranked_df["salary_gap_pct"] = pd.to_numeric(ranked_df["salary_gap_pct"], errors="coerce")
        ranked_df = ranked_df.dropna(subset=["title", "salary_gap_pct"])
        above_market_roles = ranked_df.sort_values("salary_gap_pct", ascending=True)["title"].head(3).astype(str).tolist()
        below_market_roles = ranked_df.sort_values("salary_gap_pct", ascending=False)["title"].head(3).astype(str).tolist()

    top_job_titles = title_series.value_counts().head(5).index.tolist() if not title_series.empty else []
    recruiter_share = (
        company_jobs_df["job_type_category"].fillna("unknown").eq("recruiter").mean()
        if "job_type_category" in company_jobs_df.columns and len(company_jobs_df) > 0
        else 0.0
    )

    return {
        "top_job_titles": top_job_titles,
        "seniority_mix": seniority_mix,
        "role_mix": role_mix,
        "location_mix": location_mix,
        "salary_min": salary_min,
        "salary_median": salary_median,
        "salary_max": salary_max,
        "above_market_roles": above_market_roles,
        "below_market_roles": below_market_roles,
        "recruiter_share": recruiter_share,
    }


def explain_job_pay_difference(job_row: pd.Series, company_jobs_df: pd.DataFrame) -> str:
    pay_label, _ = get_pay_vs_market_label(job_row.get("salary_gap_pct"))
    evidence = summarize_company_evidence(company_jobs_df)
    prompt = (
        "You are a business/data analyst. Explain in plain English why this job may be above or below "
        "estimated market pay. Mention that this is an estimate, not a definitive conclusion. "
        "Avoid overclaiming. Use phrases like may reflect, appears partly driven by, the data suggests, "
        "and one caveat is. Mention uncertainty when evidence is limited.\n\n"
        f"Job title: {job_row.get('title', '')}\n"
        f"Company: {format_company_name(job_row.get('company_lower'))}\n"
        f"Location: {job_row.get('location', 'unknown')}\n"
        f"Listed salary: {format_gbp(job_row.get('salary'))}\n"
        f"Estimated market salary: {format_gbp(job_row.get('predicted_salary'))}\n"
        f"Pay vs market: {pay_label}\n"
        f"Job description: {job_row.get('description', '')}\n"
        f"Top job titles at this company in the dataset: {evidence['top_job_titles']}\n"
        f"Seniority mix: {evidence['seniority_mix']}\n"
        f"Role mix: {evidence['role_mix']}\n"
        f"Location mix: {evidence['location_mix']}\n"
        f"Salary range: min {format_gbp(evidence['salary_min'])}, median {format_gbp(evidence['salary_median'])}, max {format_gbp(evidence['salary_max'])}\n"
        f"Sample above-market roles: {evidence['above_market_roles']}\n"
        f"Sample below-market roles: {evidence['below_market_roles']}\n"
        f"Recruiter-heavy share: {evidence['recruiter_share']:.2f}\n"
    )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=250,
                messages=[{"role": "user", "content": prompt}],
            )
            answer_parts = [getattr(block, "text", "") for block in response.content]
            answer_text = " ".join(part for part in answer_parts if part).strip()
            if answer_text:
                return answer_text
        except Exception as exc:
            return f"Claude request failed: {exc}"

    return (
        f"{job_row.get('title', 'This job')} at {format_company_name(job_row.get('company_lower'))} is marked as "
        f"{pay_label.lower()}. That may reflect job seniority, company brand, location, or a specialized role. "
        "This is an estimate, not a definitive conclusion."
    )


def render_company_overview_tab(company_df: pd.DataFrame) -> None:
    st.subheader("Company Overview")
    st.caption(
        "Training/provider-style listings have been excluded to focus on real job market signals."
    )
    st.write("Pay vs Market compares listed salaries with what the model expects similar jobs to pay.")

    render_top_summary(company_df)

    st.dataframe(
        build_company_display_table(company_df),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Pay vs Market": st.column_config.TextColumn(
                "Pay vs Market",
                help="Compares listed salaries with what the model expects similar jobs to pay.",
            )
        },
    )


def render_individual_jobs_tab(jobs_df: pd.DataFrame) -> None:
    st.subheader("Individual Jobs")
    st.caption(
        "Training/provider-style listings have been excluded to focus on real job market signals."
    )

    company_options = ["All"] + sorted(jobs_df["company_lower"].dropna().astype(str).unique().tolist())
    search_area_options = ["All"] + sorted(
        jobs_df["search_location"].dropna().astype(str).unique().tolist()
    )
    pay_options = ["All", "Above market", "About market", "Below market"]

    selected_company = st.selectbox("Company", options=company_options, format_func=lambda value: "All" if value == "All" else format_company_name(value))
    selected_search_area = st.selectbox("Search Area", options=search_area_options)
    selected_pay = st.selectbox("Pay vs Market", options=pay_options)

    filtered_jobs_df = jobs_df.copy()
    if selected_company != "All":
        filtered_jobs_df = filtered_jobs_df[filtered_jobs_df["company_lower"] == selected_company]
    if selected_search_area != "All":
        filtered_jobs_df = filtered_jobs_df[
            filtered_jobs_df["search_location"] == selected_search_area
        ]

    if selected_pay == "Above market":
        filtered_jobs_df = filtered_jobs_df[filtered_jobs_df["salary_gap_pct"] < -0.02]
    elif selected_pay == "About market":
        filtered_jobs_df = filtered_jobs_df[
            filtered_jobs_df["salary_gap_pct"].between(-0.02, 0.02, inclusive="both")
        ]
    elif selected_pay == "Below market":
        filtered_jobs_df = filtered_jobs_df[filtered_jobs_df["salary_gap_pct"] > 0.02]

    filtered_jobs_df = filtered_jobs_df.sort_values("salary_gap_pct", ascending=True)

    if filtered_jobs_df.empty:
        st.info("No jobs match the current filters.")
        return

    display_jobs_df = filtered_jobs_df.head(100).copy()
    if len(filtered_jobs_df) > 100:
        st.info(f"Showing the first 100 matching jobs out of {len(filtered_jobs_df)}.")

    table_event = st.dataframe(
        build_jobs_display_table(display_jobs_df),
        use_container_width=True,
        hide_index=True,
        column_config={
            "External Link": st.column_config.LinkColumn(
                "External Link",
                display_text="Open job",
                validate="^https?://.*$",
            ),
        },
        on_select="rerun",
        selection_mode="single-row",
    )

    selected_rows = table_event.selection.get("rows", []) if table_event is not None else []
    if not selected_rows:
        st.info("Select a row in the jobs table to explain that job.")
        return

    selected_job_row = display_jobs_df.iloc[int(selected_rows[0])]

    button_col_1, button_col_2 = st.columns(2)
    with button_col_1:
        if st.button("Explain this pay difference"):
            company_jobs_df = jobs_df[jobs_df["company_lower"] == selected_job_row["company_lower"]].copy()
            with st.spinner("Please wait. This explanation usually takes 5–10 seconds."):
                st.markdown(explain_job_pay_difference(selected_job_row, company_jobs_df))
    with button_col_2:
        selected_job_url = str(selected_job_row.get("redirect_url", "")).strip()
        if selected_job_url:
            st.link_button("Open job posting", selected_job_url)
        else:
            st.caption("No job link available")


def build_simple_rag_list(results: list[dict], company_df: pd.DataFrame) -> list[dict]:
    output_rows = []
    for result in results:
        company_name = result["company_lower"]
        company_row_df = company_df[company_df["company_lower"] == company_name]
        if company_row_df.empty:
            continue

        company_row = company_row_df.iloc[0]
        pay_label, _ = get_pay_vs_market_label(company_row.get("avg_salary_gap_pct"))
        hiring_label, _ = get_hiring_level(company_row.get("posting_count"))
        reviews_label, _ = get_reviews_label(
            company_row.get("external_rating"),
            company_row.get("external_review_count"),
        )
        reason = (
            f"The data suggests {pay_label.split(' ', 1)[-1].lower()} with {hiring_label.lower()}."
        )
        output_rows.append(
            {
                "company_name": format_company_name(company_name),
                "pay_label": pay_label,
                "hiring_label": hiring_label,
                "reviews_label": reviews_label,
                "reason": reason,
            }
        )
    return output_rows


def ask_market_data_with_claude(user_query: str, results: list[dict], simple_rows: list[dict]) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return ""

    context_text = "\n\n".join(
        f"Company: {result['company_lower']}\nContext: {result['document_text']}"
        for result in results
    )
    structure_text = "\n".join(
        [
            f"{index}. {row['company_name']} | {row['pay_label']} | {row['reviews_label']} | Hiring: {row['hiring_label']}"
            for index, row in enumerate(simple_rows, start=1)
        ]
    )
    prompt = (
        "You are a business/data analyst. Answer the user's actual question directly using only the retrieved context below. "
        "Do not simply restate the table. Explain why a company appears strong or weak using the evidence in the context. "
        "Mention uncertainty when review count is small or data is limited. "
        "Use plain language. Keep the answer concise but insightful. Use phrases like may reflect, appears partly driven by, the data suggests, and one caveat is. "
        "Return a short ranked list with this structure for each company: company name, pay label, hiring level, rating/review count, short reason.\n\n"
        f"User question: {user_query}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        f"Candidate company list:\n{structure_text}"
    )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=350,
            messages=[{"role": "user", "content": prompt}],
        )
        answer_parts = [getattr(block, "text", "") for block in response.content]
        return " ".join(part for part in answer_parts if part).strip()
    except Exception as exc:
        return f"Claude request failed: {exc}"


def render_ask_results(user_query: str, results: list[dict], company_df: pd.DataFrame) -> None:
    st.write("Pay shows whether companies pay above or below typical market salaries for similar roles.")
    st.markdown("### Top Companies for Data Science Roles")

    simple_rows = build_simple_rag_list(results, company_df)
    claude_answer = ask_market_data_with_claude(user_query, results, simple_rows)

    if claude_answer and not claude_answer.startswith("Claude request failed:"):
        st.markdown(claude_answer)
    else:
        for index, row in enumerate(simple_rows, start=1):
            st.markdown(
                "\n".join(
                    [
                        f"**{index}. {row['company_name']}**",
                        row["pay_label"],
                        row["reviews_label"],
                        f"Hiring: {row['hiring_label']}",
                        f"Reason: {row['reason']}",
                    ]
                )
            )
        if claude_answer:
            st.caption(claude_answer)

    retrieved_companies = [row["company_name"] for row in simple_rows]
    st.caption(f"Retrieved companies: {', '.join(retrieved_companies)}")


def render_ask_tab(company_df: pd.DataFrame) -> None:
    st.subheader("Ask the Market Data")
    with st.form("ask_market_data_form"):
        user_query = st.text_input(
            "Ask the market data",
            value="Which companies look strongest for data science roles?",
        )
        submitted = st.form_submit_button("Ask the market data")

    if submitted:
        try:
            results = retrieve_top_contexts(user_query, top_k=5)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("Run `python -m src.rag.build_rag_index` to create the RAG index first.")
            return

        render_ask_results(user_query, results, company_df)


def render_methodology_section() -> None:
    with st.expander("Methodology"):
        st.write("- Job postings are collected from the Adzuna API.")
        st.write("- Data is loaded into BigQuery.")
        st.write("- SQL creates job features.")
        st.write(
            "- A RandomForestRegressor model estimates expected salary."
        )
        st.write("- The main model inputs are simple role keywords from job titles, company, and location.")
        st.write(
            "- Search area and job type are collected for analysis and the dashboard, but they are not currently used by the salary model."
        )
        st.write(
            "- SQL also creates cleaned company and title fields plus the average listed salary target used for training."
        )
        st.write("- The dashboard compares listed salary with estimated market salary.")
        st.write("- Training/provider-style listings are filtered out.")
        st.write("- External company ratings come from Google Places API when available.")
        st.write(
            "- RAG answers retrieve the most relevant company summaries and ask Claude to answer only from that retrieved context."
        )


def render_sources_section() -> None:
    with st.expander("Data sources and refresh"):
        st.write("- Job postings: Adzuna API")
        st.write("- Data warehouse: BigQuery")
        st.write("- External ratings: Google Places API")
        st.write("- LLM explanations: Anthropic Claude")
        st.write("- Refresh: updated daily via automated pipeline")
        st.write("- Current version: batch-updated, not real-time streaming")


def main() -> None:
    st.title("UK Market Intelligence System")
    st.write(
        "This dashboard tracks a subset of UK data, analytics, and AI job postings, focused on London, "
        "Edinburgh, Glasgow, and UK-wide/remote-style searches. It is not the full UK job market."
    )
    st.write(
        "This tool helps people compare job opportunities, spot roles that may pay above or below market, and decide which applications to prioritize."
    )
    st.caption("Updated daily via an automated pipeline.")
    st.caption(get_last_updated_text())

    company_df = load_csv(get_processed_path("company_signals.csv"))
    jobs_df = prepare_jobs_dataframe(load_csv(get_processed_path("scored_jobs.csv")))
    enrichment_df = load_csv(get_processed_path("company_trust_signals.csv"), required=False)

    company_df = merge_company_data(company_df, enrichment_df)
    company_df = prepare_company_dataframe(company_df, jobs_df)

    company_tab, jobs_tab, ask_tab = st.tabs(
        ["Company Overview", "Individual Jobs", "Ask the Market Data"]
    )

    with company_tab:
        render_company_overview_tab(company_df)

    with jobs_tab:
        render_individual_jobs_tab(jobs_df)

    with ask_tab:
        render_ask_tab(company_df)

    render_methodology_section()
    render_sources_section()
    st.caption("Built by David Rauch")


if __name__ == "__main__":
    main()
