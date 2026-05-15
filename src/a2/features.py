"""Wikitext → feature dict. Pure regex; no nltk or external models.

Add features here; everything downstream consumes FEATURE_KEYS from config.
"""

from __future__ import annotations

import random
import re

import pandas as pd

from .config import CLASS_LEVEL, CLASS_ORDER, FEATURE_KEYS
from .io import read_revision

WIKILINK_RE   = re.compile(r"\[\[([^\]\|]+)")
EXTLINK_RE    = re.compile(r"\[https?://[^\] ]+")
TEMPLATE_RE   = re.compile(r"\{\{")
IMAGE_RE      = re.compile(r"(?i)\[\[(?:Image|File):")
REF_RE        = re.compile(r"<ref", re.IGNORECASE)
HEADING_RE    = re.compile(r"(?m)^(==+)\s*[^=]")
TABLE_OPEN_RE = re.compile(r"(?m)^\{\|")
LIST_RE       = re.compile(r"(?m)^[\*#]+")
BOLD_RE       = re.compile(r"'''")
ITALIC_RE     = re.compile(r"(?<!')''(?!')")
INFOBOX_RE    = re.compile(r"\{\{\s*[Ii]nfobox")
CATEGORY_RE   = re.compile(r"\[\[Category:", re.IGNORECASE)
SENTENCE_RE   = re.compile(r"[.!?]+\s")


def features(text: str | None) -> dict[str, float]:
    """Extract 21 markup/structural features from raw wikitext."""
    if not text:
        return {k: 0 for k in FEATURE_KEYS}
    words = text.split()
    n_words = len(words) or 1
    headings = HEADING_RE.findall(text)
    types = {w.lower() for w in words}
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    sentences = SENTENCE_RE.split(text)
    return {
        "chars":             len(text),
        "tokens":            len(words),
        "lines":             text.count("\n") + 1,
        "paragraphs":        len(paragraphs),
        "sentences":         len(sentences),
        "avg_word_len":      round(sum(len(w) for w in words) / n_words, 3),
        "avg_sent_len":      round(n_words / max(len(sentences), 1), 3),
        "type_token_ratio":  round(len(types) / n_words, 4),
        "wikilinks":         len(WIKILINK_RE.findall(text)),
        "extlinks":          len(EXTLINK_RE.findall(text)),
        "templates":         len(TEMPLATE_RE.findall(text)),
        "images":            len(IMAGE_RE.findall(text)),
        "refs":              len(REF_RE.findall(text)),
        "headings":          len(headings),
        "max_heading_depth": max((len(h) for h in headings), default=0),
        "tables":            len(TABLE_OPEN_RE.findall(text)),
        "list_items":        len(LIST_RE.findall(text)),
        "bold":              len(BOLD_RE.findall(text)) // 2,
        "italic":            len(ITALIC_RE.findall(text)) // 2,
        "categories":        len(CATEGORY_RE.findall(text)),
        "has_infobox":       int(bool(INFOBOX_RE.search(text))),
    }


def sample_features(df: pd.DataFrame, per_class: int, seed: int = 42) -> pd.DataFrame:
    """Stratified sample → DataFrame of features keyed by (year, label, pageid, revid)."""
    rng = random.Random(seed)
    rows = []
    for (year, label), group in df.groupby(["year", "label"]):
        idxs = list(group.index)
        rng.shuffle(idxs)
        for idx in idxs[:per_class]:
            r = df.loc[idx]
            text = read_revision(year, int(r["revid"]))
            if not text:
                continue
            f = features(text)
            f.update({
                "year": year, "label": label, "split": r["split"],
                "pageid": int(r["pageid"]), "revid": int(r["revid"]),
            })
            rows.append(f)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["label"] = pd.Categorical(out["label"], categories=CLASS_ORDER, ordered=True)
        out["level"] = out["label"].map(CLASS_LEVEL).astype("int8")
    return out


def extract_full(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for every row. Slow — for the full pipeline only."""
    rows = []
    for _, r in df.iterrows():
        year = r["year"]
        revid = int(r["revid"])
        text = read_revision(year, revid)
        f = features(text)
        f.update({
            "year": year, "label": r["label"], "split": r["split"],
            "pageid": int(r["pageid"]), "revid": revid,
        })
        rows.append(f)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["label"] = pd.Categorical(out["label"], categories=CLASS_ORDER, ordered=True)
        out["level"] = out["label"].map(CLASS_LEVEL).astype("int8")
    return out
