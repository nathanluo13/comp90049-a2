"""Stage 4 — aggregate metrics and write figures for the paper. STUB.

Reads everything in data/metrics/*.json and prints a comparison table.
Extend to produce PNGs in data/figures/ (confusion matrices, learning curves, …).

Run with:
    uv run a2 evaluate
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from ...config import METRICS, ensure_dirs


def main(args: argparse.Namespace | None = None) -> None:
    ensure_dirs()
    files = sorted(METRICS.glob("*.json"))
    if not files:
        print("evaluate: no metrics found. Run `a2 train` first.")
        return
    rows = [json.loads(f.read_text()) for f in files]
    df = pd.DataFrame(rows)
    print("evaluate: metric summary across", len(df), "runs:")
    print(df.to_string(index=False))


def add_args(p: argparse.ArgumentParser) -> None:
    pass


if __name__ == "__main__":
    main()
