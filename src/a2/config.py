"""Project-wide paths and constants. Edit here, not in scripts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "Raw"
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS = DATA / "models"
METRICS = DATA / "metrics"
FIGURES = DATA / "figures"
REPORTS = DATA / "reports"

DATASETS = {
    "2015": {
        "dir": RAW / "2015_english_wikipedia_quality_dataset",
        "label_col": "ordered_class",
        "pageid_col": "pageid",
        "revid_col": "revid",
    },
    "2017": {
        "dir": RAW / "2017_english_wikipedia_quality_dataset",
        "label_col": "rating",
        "pageid_col": "article_pageid",
        "revid_col": "article_revid",
    },
}

CLASS_ORDER = ["Stub", "Start", "C", "B", "GA", "FA"]
CLASS_LEVEL = {c: i for i, c in enumerate(CLASS_ORDER)}
CLASS_COLORS = {
    "Stub":  "#d73027",
    "Start": "#fc8d59",
    "C":     "#fee090",
    "B":     "#e0f3f8",
    "GA":    "#91bfdb",
    "FA":    "#4575b4",
}

FEATURE_KEYS = [
    "chars", "tokens", "lines", "paragraphs", "sentences",
    "avg_word_len", "avg_sent_len", "type_token_ratio",
    "wikilinks", "extlinks", "templates", "images", "refs",
    "headings", "max_heading_depth", "tables", "list_items",
    "bold", "italic", "categories", "has_infobox",
]

LABELS_PARQUET = PROCESSED / "labels.parquet"
FEATURES_PARQUET = PROCESSED / "features.parquet"


def ensure_dirs() -> None:
    for p in (PROCESSED, MODELS, METRICS, FIGURES, REPORTS):
        p.mkdir(parents=True, exist_ok=True)
