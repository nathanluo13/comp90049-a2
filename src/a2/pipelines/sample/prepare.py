"""Stage 1 — normalise both TSV schemas into a single labels parquet.

Output: data/processed/labels.parquet
Columns: year, split, pageid, revid, label, level

Run with:
    uv run a2 prepare
"""

from __future__ import annotations

import argparse
import time

from ...config import CLASS_LEVEL, LABELS_PARQUET, ensure_dirs
from ...io import load_master


def main(args: argparse.Namespace | None = None) -> None:
    ensure_dirs()
    t0 = time.time()
    df = load_master()[["year", "split", "pageid", "revid", "label"]].copy()
    df["level"] = df["label"].map(CLASS_LEVEL).astype("int8")
    LABELS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LABELS_PARQUET, index=False)
    print(f"prepare: wrote {LABELS_PARQUET.relative_to(LABELS_PARQUET.parents[2])} "
          f"({len(df):,} rows) in {time.time()-t0:.1f}s")


def add_args(p: argparse.ArgumentParser) -> None:
    pass  # no flags yet


if __name__ == "__main__":
    main()
