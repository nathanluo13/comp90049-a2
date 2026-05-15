"""Raw-data loaders. Lazy by design — never walk Raw/ in a tight loop."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import DATASETS


def load_split(year: str, split: str) -> pd.DataFrame:
    cfg = DATASETS[year]
    path = cfg["dir"] / "datasets" / f"{split}-set.tsv"
    df = pd.read_csv(path, sep="\t")
    df["year"] = year
    df["split"] = split
    df["label"] = df[cfg["label_col"]].astype(str)
    df["pageid"] = df[cfg["pageid_col"]]
    df["revid"] = df[cfg["revid_col"]]
    return df


def load_master() -> pd.DataFrame:
    return pd.concat(
        [load_split(y, s) for y in DATASETS for s in ("training", "test")],
        ignore_index=True,
    )


def revision_path(year: str, revid: int) -> Path:
    return DATASETS[year]["dir"] / "revisiondata" / str(revid)


@lru_cache(maxsize=2048)
def read_revision(year: str, revid: int) -> str | None:
    p = revision_path(year, revid)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"<<error reading {p}: {exc}>>"
