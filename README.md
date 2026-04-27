# UK Market Intelligence System

## Overview

This project analyzes UK data, analytics, and AI job postings to understand how compensation and hiring activity vary across roles and companies.

At its core, it asks a simple question:

Given a job posting, what would we expect it to pay based on similar roles, and how does the listed salary compare?

The system combines real job data, a machine learning salary estimate, and company-level aggregation to surface where jobs and companies appear to pay above or below market.

The goal is not to perfectly model the labor market. It is to provide a structured, data-driven way to compare opportunities and identify where compensation may be misaligned.

---

## Why this matters

Job markets are noisy and difficult to interpret:

- similar roles can have very different salaries  
- job titles are inconsistent across companies  
- compensation is often only partially disclosed  

This makes it hard to know whether an offer is competitive or whether a role is worth prioritizing.

By estimating expected pay and comparing it to observed salaries, this system provides a directional signal that helps:

- identify potentially underpriced roles  
- compare companies on compensation behavior  
- prioritize applications based on expected value  

---

## Approach

### 1. Job data (observed)

Job postings are collected from the Adzuna API.

The data includes:
- job title  
- company  
- location  
- listed salary (when available)  
- job description  

This forms the base dataset used throughout the system.

---

### 2. Feature construction

Raw job data is standardized and transformed into features used for modeling.

These include:
- role indicators derived from job titles  
- location and search area  
- job type classification (employer vs recruiter vs training)  
- structured salary fields  

The goal is to make roles comparable despite inconsistent formatting.

---

### 3. Salary estimation (modeled)

A machine learning model is trained to estimate expected salary based on job features.

The model learns patterns such as:
- how compensation varies by role type  
- differences across locations  
- variation across companies and job categories  

For each job:

Expected salary = model prediction  
Observed salary = listed salary  

The difference between the two becomes the core signal.

---

### 4. Pay vs market signal

Each job is labeled based on how its salary compares to the model estimate:

- above market  
- about market  
- below market  

This provides a simple, interpretable way to evaluate compensation without requiring users to interpret raw numbers.

---

### 5. Company-level aggregation

Job-level signals are aggregated to the company level.

For each company, the system estimates:
- hiring intensity (number of postings)  
- average pay position relative to market  
- salary distribution  
- mix of roles and seniority  

External signals such as ratings and review counts are added where available.

This allows comparison across companies, not just individual jobs.

---

### 6. Explanation layer (RAG + LLM)

A retrieval step selects the most relevant companies for a given query.

A language model then generates explanations using only the retrieved data, including:
- role mix  
- seniority distribution  
- salary range  
- example above- and below-market roles  

The goal is to move beyond ranking and provide reasoning grounded in the underlying data.

---

## Key observations

### 1. Compensation varies significantly within similar roles

Even after controlling for role and location, there is meaningful variation in pay across companies.

This suggests that company-level factors play a large role in compensation.

---

### 2. Role mix drives a large share of differences

Companies with a higher share of senior or specialized roles tend to appear above market.

In contrast, companies with more junior or recruiter-driven postings often appear below market.

---

### 3. Location remains a strong driver

Roles tied to higher-cost markets tend to cluster at higher salary levels.

Differences in location mix can meaningfully shift a company’s overall position.

---

### 4. Small sample sizes introduce uncertainty

Companies with few postings or limited salary data can appear extreme.

These cases require cautious interpretation.

---

## Limitations

This is a simplified system:

- only a subset of UK job postings is captured  
- salary data is incomplete for many roles  
- the model does not capture all company-specific factors  
- no longitudinal tracking of roles over time  

Results should be interpreted as directional signals, not precise estimates.

---

## Tech stack

- Python (pandas, scikit-learn)
- SQL + BigQuery
- Streamlit
- Anthropic Claude (RAG-based explanations)
- Google Places API (external ratings)

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
