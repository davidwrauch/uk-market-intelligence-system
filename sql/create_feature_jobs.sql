CREATE OR REPLACE TABLE `{project_id}.{dataset}.feature_jobs` AS
SELECT
    ingested_at,
    job_id,
    search_term,
    search_location,
    location_filter_applied,
    title,
    company,
    location,
    category,
    description,
    redirect_url,
    created,
    salary_min,
    salary_max,
    CASE
        WHEN salary_min IS NOT NULL AND salary_max IS NOT NULL THEN (salary_min + salary_max) / 2
        WHEN salary_min IS NOT NULL THEN salary_min
        WHEN salary_max IS NOT NULL THEN salary_max
        ELSE NULL
    END AS salary,
    LOWER(title) LIKE '%senior%' AS senior_flag,
    LOWER(title) LIKE '%lead%' AS lead_flag,
    LOWER(title) LIKE '%machine learning%' OR LOWER(title) LIKE '% ml %' AS ml_flag,
    LOWER(title) LIKE '%analyst%' AS analyst_flag,
    LOWER(TRIM(company)) AS company_lower,
    LOWER(TRIM(title)) AS title_lower,
    CASE
        WHEN LOWER(TRIM(company)) LIKE '%training%' THEN 'training_or_spam'
        WHEN LOWER(TRIM(company)) LIKE '%learning%' THEN 'training_or_spam'
        WHEN LOWER(TRIM(company)) LIKE '%course%' THEN 'training_or_spam'
        WHEN LOWER(TRIM(company)) LIKE '%bootcamp%' THEN 'training_or_spam'
        WHEN LOWER(TRIM(company)) LIKE '%career switch%' THEN 'training_or_spam'
        WHEN LOWER(TRIM(company)) LIKE '%itonlinelearning%' THEN 'training_or_spam'
        WHEN LOWER(TRIM(company)) LIKE '%newto training%' THEN 'training_or_spam'
        WHEN LOWER(TRIM(company)) LIKE '%recruit%' THEN 'recruiter'
        WHEN LOWER(TRIM(company)) LIKE '%career%' THEN 'recruiter'
        ELSE 'employer'
    END AS job_type_category,
    CASE
        WHEN LOWER(TRIM(company)) LIKE '%training%' THEN FALSE
        WHEN LOWER(TRIM(company)) LIKE '%learning%' THEN FALSE
        WHEN LOWER(TRIM(company)) LIKE '%course%' THEN FALSE
        WHEN LOWER(TRIM(company)) LIKE '%bootcamp%' THEN FALSE
        WHEN LOWER(TRIM(company)) LIKE '%career switch%' THEN FALSE
        WHEN LOWER(TRIM(company)) LIKE '%itonlinelearning%' THEN FALSE
        WHEN LOWER(TRIM(company)) LIKE '%newto training%' THEN FALSE
        ELSE TRUE
    END AS is_valid_job
FROM `{project_id}.{dataset}.raw_jobs`;
