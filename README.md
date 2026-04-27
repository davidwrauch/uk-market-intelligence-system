# UK Market Intelligence System

This is a small project I built to make job hunting a little less opaque.

It looks at real UK job postings in data, analytics, and AI and tries to answer a simple question: is this job paying what it should?

Behind the scenes, it pulls job listings, builds a simple model to estimate what similar roles tend to pay, and then compares that to the salary being offered. On top of that, it rolls things up to the company level so you can see which companies tend to pay more or less, and how actively they are hiring.

There’s also a layer that lets you ask questions about the market. It pulls the most relevant companies and uses an LLM to explain what might be going on in plain English, based only on the data in the system.

The goal isn’t to be perfect. It’s to give you a directional signal so you can decide which roles to focus on, which ones might be underpaying, and where you might have more leverage.

## What it’s doing under the hood

Job postings come from the Adzuna API and are stored and shaped in BigQuery. A simple machine learning model estimates expected salary based on things like role, location, and other features derived from the posting. The app then compares that estimate to the listed salary.

Company ratings come from Google Places when they’re available. The explanation layer uses retrieval plus Claude so that answers are grounded in the actual companies pulled from the data.

The app itself is built in Streamlit and is meant to feel lightweight and interactive.

## Running it locally

Install dependencies and run:

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
