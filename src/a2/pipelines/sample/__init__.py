"""Sample pipeline — the worked example.

Stages:
    prepare    TSV → labels.parquet
    extract    wikitext → features.parquet
    train      fit a model (STUB — replace with real one)
    evaluate   aggregate metrics from data/metrics/*.json
    report     fast static HTML EDA → data/reports/eda.html
    explainer  long-form HTML walkthrough → data/reports/explainer.html

Run individual stages:
    uv run a2 sample prepare
    uv run a2 sample extract --sample 200
    uv run a2 sample report

Run the EDA-only subset end-to-end:
    uv run a2 sample all
"""
