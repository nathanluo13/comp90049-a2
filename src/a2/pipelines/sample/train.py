"""Stage 3 — fit models. STUB; teams to flesh out per research question.

Convention:
- Read data/processed/features.parquet (or a --features-path override).
- Persist trained models to data/models/<rq>-<model>.joblib.
- Persist per-model metrics JSON to data/metrics/<rq>-<model>.json.

Run with:
    uv run a2 train --rq rq1-structural --model lr
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from ...config import (FEATURE_KEYS, FEATURES_PARQUET, METRICS, MODELS,
                      ensure_dirs)


def load_features(path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing {path}. Run `a2 extract` first.")
    return pd.read_parquet(path)


def main(args: argparse.Namespace | None = None) -> None:
    ensure_dirs()
    feats_path = FEATURES_PARQUET
    if args and args.features_path:
        from pathlib import Path
        feats_path = Path(args.features_path)
    df = load_features(feats_path)

    t0 = time.time()
    # ---------------------------------------------------------------
    # STUB: replace with your model. The block below is a placeholder
    # that simply records dataset shape so the pipeline runs end-to-end.
    # ---------------------------------------------------------------
    train = df[df["split"] == "training"]
    test = df[df["split"] == "test"]
    feature_matrix = train[FEATURE_KEYS]

    metrics = {
        "rq": args.rq if args else "unset",
        "model": args.model if args else "stub",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(feature_matrix.shape[1]),
        "fit_seconds": round(time.time() - t0, 3),
    }

    tag = f"{metrics['rq']}-{metrics['model']}"
    out = METRICS / f"{tag}.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"train: wrote {out.name} (stub) — replace this with real model fitting.")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rq", default="rq1",
                   help="research-question tag for organising outputs")
    p.add_argument("--model", default="stub",
                   help="model name (lr, rf, gbm, mlp, …)")
    p.add_argument("--features-path", default=None,
                   help="override the features parquet location")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    add_args(ap)
    main(ap.parse_args())
