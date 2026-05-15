"""Stage 2 — read each revision file and compute features.

Input:  data/processed/labels.parquet
Output: data/processed/features.parquet

Slow on the full corpus (~63k revisions); use --sample for development.

Run with:
    uv run a2 extract                  # full corpus
    uv run a2 extract --sample 500     # 500 rows per (year, label) for iteration
    uv run a2 extract --year 2015      # restrict to one year
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from ...config import FEATURES_PARQUET, LABELS_PARQUET, ensure_dirs
from ...features import extract_full, sample_features


def main(args: argparse.Namespace | None = None) -> None:
    ensure_dirs()
    if not LABELS_PARQUET.exists():
        raise SystemExit(f"missing {LABELS_PARQUET}. Run `a2 prepare` first.")

    labels = pd.read_parquet(LABELS_PARQUET)
    if args and args.year:
        labels = labels[labels["year"] == args.year].reset_index(drop=True)
    if args and args.split:
        labels = labels[labels["split"] == args.split].reset_index(drop=True)

    t0 = time.time()
    if args and args.sample:
        feats = sample_features(labels, per_class=args.sample)
        out_path = FEATURES_PARQUET.with_name(f"features-sample{args.sample}.parquet")
    else:
        print(f"extract: computing features on {len(labels):,} rows… (this is slow)")
        feats = extract_full(labels)
        out_path = FEATURES_PARQUET

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    print(f"extract: wrote {out_path.name} ({len(feats):,} rows, "
          f"{len(feats.columns)} cols) in {time.time()-t0:.1f}s")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sample", type=int, default=None,
                   help="per (year, label) sample size (development mode)")
    p.add_argument("--year", choices=["2015", "2017"], default=None)
    p.add_argument("--split", choices=["training", "test"], default=None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    add_args(ap)
    main(ap.parse_args())
