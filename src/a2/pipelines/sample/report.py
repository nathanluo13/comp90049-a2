"""Stage 5a — fast static EDA report.

Output: data/reports/eda.html (or --out)

Run with:
    uv run a2 report
    uv run a2 report --sample 300
"""

from __future__ import annotations

import argparse
import html
import random
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio

from ...config import (CLASS_COLORS, CLASS_ORDER, DATASETS, FEATURE_KEYS,
                      REPORTS, ensure_dirs)
from ...features import sample_features
from ...io import load_master, read_revision


def _fig_html(fig) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       config={"displayModeBar": False})


def _kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="v">{value}</div><div class="k">{label}</div></div>'


def _samples(master: pd.DataFrame) -> str:
    rng = random.Random(7)
    blocks = []
    for year in DATASETS:
        for cls in CLASS_ORDER:
            sub = master[(master["year"] == year)
                         & (master["split"] == "training")
                         & (master["label"] == cls)]
            if sub.empty:
                continue
            idxs = list(sub.index)
            rng.shuffle(idxs)
            row = sub.loc[idxs[0]]
            text = read_revision(year, int(row["revid"]))
            if not text:
                continue
            snippet = text[:1500] + ("…" if len(text) > 1500 else "")
            blocks.append(
                f'<div class="sample"><div class="meta">[{year} · {cls}] '
                f'pageid={int(row["pageid"])} · revid={int(row["revid"])} '
                f'· {len(text):,} chars</div>'
                f'<pre>{html.escape(snippet)}</pre></div>'
            )
    return "\n".join(blocks)


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
       max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #222; }
h1 { border-bottom: 2px solid #333; padding-bottom: .3em; }
h2 { margin-top: 2em; border-bottom: 1px solid #ccc; padding-bottom: .2em; }
.kpis { display: flex; gap: 1em; flex-wrap: wrap; margin: 1em 0; }
.kpi { flex: 1 1 180px; background: #f4f6f8; padding: 1em; border-radius: 8px;
       border-left: 4px solid #4575b4; }
.kpi .v { font-size: 1.6em; font-weight: 600; }
.kpi .k { color: #666; font-size: .85em; }
table { border-collapse: collapse; margin: 1em 0; font-size: .9em; width: 100%; }
th, td { padding: 6px 12px; border: 1px solid #ddd; text-align: right; }
th { background: #f4f6f8; text-align: left; }
td:first-child { text-align: left; }
.sample { background: #fafafa; border: 1px solid #ddd; border-radius: 6px;
          padding: .8em 1em; margin: .8em 0; }
.sample .meta { font-family: monospace; color: #555; font-size: .85em; margin-bottom: .4em; }
.sample pre { white-space: pre-wrap; word-wrap: break-word; max-height: 240px;
              overflow-y: auto; background: #fff; padding: .6em; border: 1px solid #eee;
              font-size: .8em; margin: 0; }
.muted { color: #777; font-size: .9em; }
"""


def build_html(master: pd.DataFrame, stats: pd.DataFrame, per_class: int) -> str:
    pivot = master.groupby(["year", "split"]).size().unstack(fill_value=0)
    pivot["total"] = pivot.sum(axis=1)

    counts = master.groupby(["year", "split", "label"]).size().reset_index(name="n")
    fig_dist = px.bar(
        counts, x="label", y="n", color="label",
        facet_col="split", facet_row="year",
        category_orders={"label": CLASS_ORDER, "split": ["training", "test"]},
        color_discrete_map=CLASS_COLORS, text="n",
    )
    fig_dist.update_layout(showlegend=False, height=480,
                           margin=dict(t=40, b=10, l=10, r=10),
                           title="Class counts — year × split")

    feat_blocks = []
    if not stats.empty:
        for m, log in [("chars", True), ("refs", False), ("templates", False),
                       ("wikilinks", True), ("headings", False)]:
            if m in stats.columns:
                fig = px.box(stats, x="label", y=m, color="label", facet_col="year",
                             category_orders={"label": CLASS_ORDER},
                             color_discrete_map=CLASS_COLORS, points="outliers",
                             title=f"{m} per article — sampled")
                if log:
                    fig.update_yaxes(type="log")
                fig.update_layout(showlegend=False, height=400, margin=dict(t=40, b=10))
                feat_blocks.append(_fig_html(fig))

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>A2 EDA</title>
<style>{CSS}</style>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head><body>
<h1>COMP90049 A2 — EDA</h1>
<p class="muted">Generated {time.strftime('%Y-%m-%d %H:%M')} · per-class sample = {per_class}</p>

<div class="kpis">
  {_kpi(f"{len(master):,}", "Label rows")}
  {_kpi(f"{(master['split']=='training').sum():,}", "Train rows")}
  {_kpi(f"{(master['split']=='test').sum():,}", "Test rows")}
  {_kpi(f"{len(stats):,}", "Sampled for feature stats")}
</div>

<h2>Year × split</h2>
{pivot.to_html(classes="data", border=0)}

<h2>Class distributions</h2>
{_fig_html(fig_dist)}

<h2>Selected feature distributions</h2>
{''.join(feat_blocks)}

<h2>Raw article samples</h2>
{_samples(master)}
</body></html>"""


def main(args: argparse.Namespace | None = None) -> None:
    ensure_dirs()
    per_class = args.sample if args and args.sample else 150
    out = Path(args.out) if args and args.out else REPORTS / "eda.html"

    t0 = time.time()
    master = load_master()
    train = master[master["split"] == "training"].reset_index(drop=True)
    stats = sample_features(train, per_class)
    out.write_text(build_html(master, stats, per_class), encoding="utf-8")
    print(f"report: wrote {out.relative_to(out.parents[2])} "
          f"({out.stat().st_size/1024:.0f} KB) in {time.time()-t0:.1f}s")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sample", type=int, default=150)
    p.add_argument("--out", default=None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    add_args(ap)
    main(ap.parse_args())
