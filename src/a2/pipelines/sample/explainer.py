"""Stage 5b — long-form HTML explainer tied to the A2 brief.

Output: data/reports/explainer.html (or --out)

Run with:
    uv run a2 explainer
    uv run a2 explainer --sample 300
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
from ...features import features, sample_features
from ...io import load_master, read_revision


CLASS_DEFS = [
    ("Stub",  "Stub",            "A few sentences. Barely useful as an article.",  "100–500 chars"),
    ("Start", "Start-class",     "Some content but obviously incomplete or unbalanced.", "1–3 KB"),
    ("C",     "C-class",         "Substantial but missing important material.",     "3–10 KB"),
    ("B",     "B-class",         "Mostly complete, well-referenced, minor issues.", "10–30 KB"),
    ("GA",    "Good Article",    "Passed a formal GA review — solid quality.",      "20–60 KB"),
    ("FA",    "Featured Article","Passed FA review — Wikipedia's highest grade.",   "40–120 KB"),
]

FEATURE_DICT = [
    ("chars",            "Length",  "Total character count of raw wikitext.",
     "Strongest single signal. Higher quality = much longer articles.",
     "Increases monotonically with class. FA medians can be 100×+ Stub."),
    ("tokens",           "Length",  "Whitespace-tokenised words.",
     "Same as chars but ratio with chars hints at language vs markup density.",
     "Monotonic increase."),
    ("lines",            "Length",  "Newline count.",
     "Caps structural size.",
     "Monotonic but dominated by sections + lists in higher classes."),
    ("paragraphs",       "Length",  "Blocks separated by blank lines.",
     "Proxy for prose density.",
     "Rises with class then plateaus around B/GA/FA."),
    ("sentences",        "Length",  "Approximate sentence count via [.!?] splits.",
     "Rough but useful — pairs with avg_sent_len.",
     "Rises with class."),
    ("avg_word_len",     "Lexical", "Mean characters per token.",
     "Slight signal — formal articles use longer Latin-derived words.",
     "Weak positive correlation."),
    ("avg_sent_len",     "Lexical", "tokens / sentences.",
     "Stylistic — well-edited prose has consistent sentence length.",
     "Mostly noise alone; useful with other features."),
    ("type_token_ratio", "Lexical", "Unique tokens / total tokens.",
     "Higher = more vocabulary diversity.",
     "DROPS with class because long articles repeat function words; this inversion is informative."),
    ("wikilinks",        "Markup",  "[[…]] openings (internal Wikipedia links).",
     "Strong signal — better articles link more concepts.",
     "Increases sharply; FA articles often have 100s."),
    ("extlinks",         "Markup",  "Bare [https://…] external links.",
     "Indicator of source attribution outside ref tags.",
     "Increases with class."),
    ("templates",        "Markup",  "{{…}} openings (transclusions).",
     "Infoboxes, navboxes, citation templates etc.",
     "Increases steeply with class; FA articles use dozens."),
    ("images",           "Markup",  "Image and file embeds.",
     "Visual richness.",
     "Stub/Start have ~0; B/GA/FA have several."),
    ("refs",             "Markup",  "<ref> tags (citations).",
     "GA/FA require strong sourcing; very discriminative for top classes.",
     "Near 0 for Stub; tens to hundreds for FA."),
    ("headings",         "Structure","Lines starting with `==`.",
     "Section count.",
     "Stubs have 0–1; FA has 6–15."),
    ("max_heading_depth","Structure","Deepest heading level (=, ==, ===, …).",
     "Captures hierarchical structure.",
     "Stubs flat (2); FA can reach 4–5."),
    ("tables",           "Structure","`{|` opens (wikitable starts).",
     "Statistics, results, etc.",
     "Most evident in GA/FA on scientific or sports topics."),
    ("list_items",       "Structure","Lines starting with `*` or `#`.",
     "Bulleted/numbered list density.",
     "Rises modestly; lower classes use ad-hoc bullets, higher classes use prose."),
    ("bold",             "Style",   "Pairs of `'''` (bold markers).",
     "Disambiguation bolds + key terms.",
     "Weak positive correlation."),
    ("italic",           "Style",   "Pairs of `''` (italic markers).",
     "Titles of works etc.",
     "Weak positive correlation."),
    ("categories",       "Meta",    "[[Category:…]] links at the bottom.",
     "Categorisation effort.",
     "Increases with class."),
    ("has_infobox",      "Structure","Whether `{{Infobox` appears.",
     "Binary — does the article have an infobox at all?",
     "Stub:~10%, FA:~95%."),
]

RESEARCH_QUESTIONS = [
    ("Can we predict an article's quality class from purely structural features (no text content)?",
     "Uses only counts of templates, refs, images, headings, etc. — interpretable and "
     "fast. Directly answers whether 'editing effort signals' alone capture quality."),
    ("How much additional accuracy comes from text content (TF-IDF, embeddings) over structural features?",
     "Sets up a clean ablation: structural baseline → +unigram TF-IDF → +BERT embeddings."),
    ("How stable is article quality across two years? Can a model trained on 2015 predict 2017 labels for the same pageids?",
     "Tests temporal generalisation — train on 2015, evaluate on 2017 shared pageids "
     "with the 2017 label as ground truth."),
    ("Is the ordinal structure (Stub < Start < C < B < GA < FA) better captured by ordinal regression than by flat classification?",
     "Compares macro-F1 of multiclass softmax vs ordinal logistic regression / regression-then-bin."),
    ("Which classes are systematically confused, and does the confusion match human-rater disagreement reported in the Wikipedia QA literature?",
     "Driven by the confusion matrix; ties into Discussion."),
]

MODELS = [
    ("Logistic Regression (multinomial)", "Classic",
     "Linear baseline. High bias, low variance. Tune C."),
    ("Random Forest",                     "Classic",
     "Mid bias / mid variance. Non-linear feature interactions. Tune n_estimators + max_depth."),
    ("Gradient Boosting (sklearn or XGBoost)", "Classic",
     "Lower bias. Often strongest non-NN model on tabular data. Tune learning_rate + n_estimators."),
    ("Linear SVM",                        "Classic",
     "Strong with TF-IDF text. Tune C."),
    ("MLP (feed-forward neural network)", "Neural",
     "Required NN. Use on combined structural + TF-IDF features. Tune hidden_layer_sizes."),
    ("DistilBERT fine-tune (optional 4-person extension)", "Neural",
     "Process the lead section only (~512 tokens). Heavier — only with GPU and time."),
]


def _fig_html(fig) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       config={"displayModeBar": False})


def _df_table(df: pd.DataFrame, idx: bool = True) -> str:
    return df.to_html(classes="data", border=0, index=idx,
                      float_format=lambda v: f"{v:,.2f}" if isinstance(v, float) else f"{v:,}")


def _kpi(value: str, label: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi"><div class="v">{value}</div>'
            f'<div class="k">{label}</div>{sub_html}</div>')


def _esc(t: str) -> str:
    return html.escape(t)


def _class_dist_fig(master):
    counts = master.groupby(["year", "split", "label"]).size().reset_index(name="n")
    fig = px.bar(counts, x="label", y="n", color="label",
                 facet_col="split", facet_row="year",
                 category_orders={"label": CLASS_ORDER, "split": ["training", "test"]},
                 color_discrete_map=CLASS_COLORS, text="n")
    fig.update_layout(showlegend=False, height=520,
                      margin=dict(t=40, b=10, l=10, r=10),
                      title="Class counts — year × split")
    return fig


def _proportion_compare(master):
    train = master[master["split"] == "training"]
    counts = train.groupby(["year", "label"]).size().reset_index(name="n")
    counts["pct"] = counts.groupby("year")["n"].transform(lambda s: s / s.sum())
    fig = px.bar(counts, x="label", y="pct", color="year", barmode="group",
                 category_orders={"label": CLASS_ORDER},
                 text=counts["pct"].apply(lambda v: f"{v:.1%}"),
                 title="Training-set class proportions — 2015 vs 2017")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=400, margin=dict(t=40, b=10))
    return fig


def _feature_box(stats, metric, log=False, title=""):
    fig = px.box(stats, x="label", y=metric, color="label", facet_col="year",
                 category_orders={"label": CLASS_ORDER},
                 color_discrete_map=CLASS_COLORS, points="outliers",
                 title=title or f"{metric} per article — by class and year")
    if log:
        fig.update_yaxes(type="log")
    fig.update_layout(showlegend=False, height=440, margin=dict(t=40, b=10))
    return fig


def _feature_correlation(stats):
    num = stats[FEATURE_KEYS + ["level"]].copy().apply(pd.to_numeric, errors="coerce")
    corr = num.corr()["level"].drop("level").sort_values()
    fig = px.bar(corr, x=corr.values, y=corr.index, orientation="h",
                 color=corr.values, color_continuous_scale="RdBu_r",
                 range_color=[-1, 1],
                 title="Pearson correlation between each feature and ordinal class level (0=Stub … 5=FA)")
    fig.update_layout(height=620, margin=dict(t=40, b=10),
                      coloraxis_showscale=False,
                      xaxis_title="correlation with quality level", yaxis_title="")
    return fig


def _transition_heatmap(master):
    a = master[master["year"] == "2015"][["pageid", "label"]].drop_duplicates("pageid")
    b = master[master["year"] == "2017"][["pageid", "label"]].drop_duplicates("pageid")
    j = a.merge(b, on="pageid", suffixes=("_2015", "_2017"))
    mat = (j.groupby(["label_2015", "label_2017"]).size()
           .unstack(fill_value=0).reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0))
    fig = px.imshow(mat, text_auto=True, aspect="auto",
                    labels=dict(x="2017 label", y="2015 label", color="pages"),
                    color_continuous_scale="Blues",
                    title=f"Label transitions for {len(j):,} shared pageids (2015 → 2017)")
    fig.update_layout(height=520, margin=dict(t=40, b=10))
    same = int((j["label_2015"] == j["label_2017"]).sum())
    return fig, mat, same, len(j)


def _quality_definitions_html() -> str:
    rows = []
    for cls, name, defn, size in CLASS_DEFS:
        color = CLASS_COLORS[cls]
        rows.append(
            f'<tr><td><span class="cls-tag" style="background:{color};">{cls}</span></td>'
            f'<td><b>{name}</b></td><td>{defn}</td><td class="muted">{size}</td></tr>'
        )
    return ('<table class="data"><thead><tr><th>Class</th><th>Full name</th>'
            '<th>Meaning</th><th>Typical size</th></tr></thead><tbody>'
            + "\n".join(rows) + "</tbody></table>")


def _feature_cards_html() -> str:
    by_group: dict[str, list] = {}
    for name, group, desc, signal, behavior in FEATURE_DICT:
        by_group.setdefault(group, []).append((name, desc, signal, behavior))
    out = []
    for group, items in by_group.items():
        cards = []
        for name, desc, signal, behavior in items:
            cards.append(
                f'<div class="card"><h4><span class="inline-code">{name}</span></h4>'
                f'<p style="margin:.2em 0"><b>What:</b> {desc}</p>'
                f'<p style="margin:.2em 0"><b>Signal:</b> {signal}</p>'
                f'<p style="margin:.2em 0;color:#59636e"><b>Expected per class:</b> {behavior}</p>'
                f'</div>'
            )
        out.append(f'<h3>{group} features</h3><div class="cards">{"".join(cards)}</div>')
    return "\n".join(out)


def _proposed_csv_html(stats) -> str:
    if stats.empty:
        return "<p>(sample empty)</p>"
    sample_rows = (stats.groupby("label", observed=True)
                   .apply(lambda g: g.head(1), include_groups=False)
                   .reset_index(level=0))
    cols = (["year", "split", "pageid", "revid", "label", "level"] + FEATURE_KEYS)
    cols = [c for c in cols if c in sample_rows.columns]
    return _df_table(sample_rows[cols].reset_index(drop=True), idx=False)


def _samples_html(master, max_chars=1200) -> str:
    rng = random.Random(11)
    blocks = []
    for cls in CLASS_ORDER:
        for year in DATASETS:
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
            stat = features(text)
            snippet = text[:max_chars] + ("…" if len(text) > max_chars else "")
            color = CLASS_COLORS[cls]
            blocks.append(
                f'<div class="sample"><div class="meta">'
                f'<span class="cls-tag" style="background:{color};">{cls}</span> '
                f'· {year} · pageid={int(row["pageid"])} · revid={int(row["revid"])} '
                f'· {stat["chars"]:,} chars · {stat["tokens"]:,} tokens '
                f'· {stat["wikilinks"]} wikilinks · {stat["refs"]} refs '
                f'· {stat["templates"]} templates · {stat["images"]} images'
                f'</div><pre>{_esc(snippet)}</pre></div>'
            )
    return "\n".join(blocks)


def _findings(master, stats, mat, same, shared):
    findings = []
    findings.append(
        f"<b>Volume:</b> {len(master):,} labelled examples across two years and two splits — "
        "comfortably past the 'tens of thousands of instances' threshold the A2 spec asks for."
    )
    train = master[master["split"] == "training"]
    c15 = train[train["year"] == "2015"]["label"].value_counts()
    c17 = train[train["year"] == "2017"]["label"].value_counts()
    findings.append(
        f"<b>Class balance:</b> 2015 training {c15.min():,}–{c15.max():,} per class; "
        f"2017 training {c17.min():,}–{c17.max():,}. Both stratified — FA slightly under-represented."
    )
    if not stats.empty:
        med = stats.groupby(["year", "label"], observed=True)["chars"].median().unstack(0)
        if "Stub" in med.index and "FA" in med.index:
            stub, fa = med.loc["Stub"].mean(), med.loc["FA"].mean()
            findings.append(
                f"<b>Length explosion:</b> median Stub ~{stub:,.0f} chars; "
                f"median FA ~{fa:,.0f} chars (~{fa/max(stub,1):.0f}× larger). "
                "Length alone is a very strong baseline signal."
            )
        refs = stats.groupby("label", observed=True)["refs"].median().reindex(CLASS_ORDER).dropna()
        if "FA" in refs and "Stub" in refs:
            findings.append(
                f"<b>Citation depth:</b> median &lt;ref&gt; tags climb "
                f"{refs['Stub']:.0f} (Stub) → {refs['C']:.0f} (C) → "
                f"{refs['GA']:.0f} (GA) → {refs['FA']:.0f} (FA)."
            )
    findings.append(
        "<b>Schema drift:</b> 2015 has <span class='inline-code'>pageid, revid, ordered_class</span>; "
        "2017 adds <span class='inline-code'>talk_pageid, talk_revid</span> and renames the label to "
        "<span class='inline-code'>rating</span>. A unified loader normalises both."
    )
    if shared:
        findings.append(
            f"<b>Temporal overlap:</b> {shared:,} pageids in both years; "
            f"{same:,} ({same/shared:.1%}) keep the same label. Natural temporal-generalisation RQ."
        )
    findings.append(
        "<b>Storage shape:</b> Label TSVs are tiny; the bulk (1.4 GB) is wikitext — "
        "63k blobs, one per revid. Always index by revid; never walk the directory."
    )
    return findings


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1240px; margin: 0 auto; padding: 2em 1.4em; color: #1f2328;
       line-height: 1.55; }
h1 { font-size: 2.1em; border-bottom: 3px solid #1f2328; padding-bottom: .25em; }
h2 { font-size: 1.55em; margin-top: 2.4em; border-bottom: 1px solid #d0d7de; padding-bottom: .2em; }
h3 { font-size: 1.15em; margin-top: 1.8em; color: #1c2c4c; }
p, li { font-size: 0.97em; }
.muted { color: #59636e; font-size: .9em; }
.kpis { display: flex; gap: .8em; flex-wrap: wrap; margin: 1.2em 0; }
.kpi { flex: 1 1 180px; background: #f6f8fa; padding: 1em 1.1em; border-radius: 10px;
       border-left: 4px solid #4575b4; }
.kpi .v { font-size: 1.55em; font-weight: 600; line-height: 1.1; }
.kpi .k { color: #59636e; font-size: .85em; margin-top: .2em; }
table.data { border-collapse: collapse; margin: 1em 0; font-size: .88em; width: 100%; }
table.data th, table.data td { padding: 6px 10px; border: 1px solid #d0d7de; }
table.data th { background: #f6f8fa; text-align: left; }
table.data td:not(:first-child) { text-align: right; }
.tree { font-family: 'SF Mono', Menlo, monospace; font-size: .85em;
        background: #0d1117; color: #c9d1d9; padding: 1em; border-radius: 8px;
        white-space: pre; overflow-x: auto; }
.callout { background: #fff8e1; border-left: 4px solid #ffb300; padding: .8em 1em;
           border-radius: 6px; margin: 1em 0; }
.callout.ok { background: #e8f5e9; border-left-color: #2e7d32; }
.callout.info { background: #e3f2fd; border-left-color: #1565c0; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
         gap: 1em; margin: 1em 0; }
.card { background: #f6f8fa; border-radius: 10px; padding: 1em 1.1em; }
.card h4 { margin: 0 0 .4em 0; font-size: 1.05em; }
.card .tag { display: inline-block; background: #1f2328; color: white; border-radius: 999px;
             padding: 0 8px; font-size: .7em; margin-left: 6px; vertical-align: middle; }
.code { background: #0d1117; color: #c9d1d9; padding: 1em; border-radius: 8px;
        font-family: 'SF Mono', Menlo, monospace; font-size: .82em;
        overflow-x: auto; white-space: pre; }
.inline-code { background: #f0f1f2; padding: 1px 5px; border-radius: 4px;
               font-family: 'SF Mono', Menlo, monospace; font-size: .87em; }
.cls-tag { display: inline-block; padding: 2px 8px; border-radius: 999px;
           font-size: .78em; font-weight: 600; }
.sample { background: #fafbfc; border: 1px solid #d0d7de; border-radius: 8px;
          padding: .8em 1em; margin: .9em 0; }
.sample .meta { font-family: 'SF Mono', Menlo, monospace; color: #59636e;
                font-size: .82em; margin-bottom: .4em; }
.sample pre { white-space: pre-wrap; word-wrap: break-word; max-height: 260px;
              overflow-y: auto; background: #fff; padding: .6em; border: 1px solid #eaeef2;
              font-size: .78em; margin: 0; line-height: 1.45; }
.toc { background: #f6f8fa; border-radius: 8px; padding: 1em 1.4em; }
.toc ol { margin: 0; padding-left: 1.4em; }
.findings li { margin: .4em 0; }
"""


def build_html(master, stats, mat, same, shared, per_class) -> str:
    n15_rev = sum(1 for _ in (DATASETS["2015"]["dir"] / "revisiondata").iterdir())
    n17_rev = sum(1 for _ in (DATASETS["2017"]["dir"] / "revisiondata").iterdir())
    n_train = int((master["split"] == "training").sum())
    n_test = int((master["split"] == "test").sum())

    pivot = master.groupby(["year", "split"]).size().unstack(fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    class_pivot = (master.groupby(["year", "split", "label"]).size()
                   .unstack("label", fill_value=0)
                   .reindex(columns=CLASS_ORDER, fill_value=0))

    fig_dist = _fig_html(_class_dist_fig(master))
    fig_cmp = _fig_html(_proportion_compare(master))
    fig_trans = _fig_html(_transition_heatmap(master)[0])

    feat_figs = []
    if not stats.empty:
        feat_figs.append(_fig_html(_feature_correlation(stats)))
        for m, log, title in [
            ("chars", True, "Article length (log) — order-of-magnitude jump across quality"),
            ("refs", False, "Citation count (<ref>) — near-zero for Stub, dozens-hundreds for FA"),
            ("templates", False, "Template uses ({{ }})"),
            ("wikilinks", True, "Internal Wikipedia links ([[ ]])"),
            ("headings", False, "Section headings (== … ==)"),
            ("type_token_ratio", False, "Vocabulary diversity (TTR)"),
        ]:
            if m in stats.columns:
                feat_figs.append(_fig_html(_feature_box(stats, m, log=log, title=title)))

    findings = _findings(master, stats, mat, same, shared)
    findings_html = "\n".join(f"<li>{f}</li>" for f in findings)

    rq_cards = "\n".join(
        f'<div class="card"><h4>RQ {i+1}</h4><p><b>{_esc(q)}</b></p>'
        f'<p class="muted">{_esc(r)}</p></div>'
        for i, (q, r) in enumerate(RESEARCH_QUESTIONS)
    )
    model_rows = "\n".join(
        f'<tr><td><b>{_esc(n)}</b></td><td>{_esc(g)}</td><td>{_esc(notes)}</td></tr>'
        for n, g, notes in MODELS
    )

    schema_15 = ("pageid    int    Wikipedia page identifier\n"
                 "revid     int    Specific revision identifier (one wikitext snapshot)\n"
                 "ordered_class    str    Quality grade — Stub | Start | C | B | GA | FA")
    schema_17 = ("article_pageid    int    Article page identifier\n"
                 "article_revid     int    Revision id for the article body\n"
                 "talk_pageid       int    Page id of the article's Talk page\n"
                 "talk_revid        int    Revision id of the matching talk page snapshot\n"
                 "rating            str    Quality grade — Stub | Start | C | B | GA | FA")

    csv_code = ("# Build a single unified DataFrame for modelling\n"
                "from a2.io import load_master\n"
                "from a2.features import extract_full\n\n"
                "labels = load_master()                          # 61,907 rows\n"
                "features_df = extract_full(labels)              # 21 engineered features\n"
                "features_df.to_parquet('data/processed/features.parquet')")

    samples_html = _samples_html(master)

    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>COMP90049 A2 — Data Explainer</title>
<style>{CSS}</style>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head><body>

<h1>COMP90049 Assignment 2 — Data Explainer</h1>
<p class="muted">Generated {time.strftime('%Y-%m-%d %H:%M')} · per-class feature sample = {per_class} revisions ·
total {len(stats):,} sampled articles</p>

<div class="toc"><b>Contents</b>
<ol>
  <li>What we have, at a glance</li>
  <li>How this maps to the A2 rubric</li>
  <li>Filesystem anatomy</li>
  <li>Raw TSV schemas</li>
  <li>Wikipedia quality classes</li>
  <li>Proposed unified CSV for modelling</li>
  <li>Feature dictionary</li>
  <li>What the features actually look like (charts)</li>
  <li>Raw article samples per class</li>
  <li>Year overlap — 2015 ↔ 2017</li>
  <li>Suggested research questions</li>
  <li>Suggested model line-up</li>
  <li>Validation strategy</li>
  <li>Risks &amp; gotchas</li>
</ol></div>

<h2>1. What we have, at a glance</h2>
<div class="kpis">
  {_kpi(f"{len(master):,}", "Labelled rows", "across 2015 + 2017, train + test")}
  {_kpi(f"{n_train:,}", "Training rows", "the ones you'll fit on")}
  {_kpi(f"{n_test:,}", "Test rows", "held out per year")}
  {_kpi(f"{n15_rev:,} + {n17_rev:,}", "Revision files", "639 MB + 727 MB raw wikitext")}
  {_kpi(f"{shared:,}", "Shared pageids", "same article in both years")}
  {_kpi(f"{same/shared:.0%}" if shared else "—", "Unchanged across years",
        f"{same:,} of {shared:,} kept same label")}
</div>
<h3>Key findings</h3>
<ul class="findings">{findings_html}</ul>

<h2>2. How this maps to the A2 rubric</h2>
<table class="data">
<thead><tr><th>A2 requirement</th><th>How this dataset satisfies it</th></tr></thead>
<tbody>
<tr><td>Tens of thousands of instances</td><td>61,911 labelled rows — well past the bar.</td></tr>
<tr><td>Complex feature space / engineering opportunity</td>
    <td>Each row points to a full raw-wikitext document. The feature space is whatever you engineer.</td></tr>
<tr><td>≥1 pre-processing / feature construction step</td>
    <td>Mandatory: wikitext → structured features (see §6).</td></tr>
<tr><td>≥2 research questions (3-person)</td><td>Five candidates in §11.</td></tr>
<tr><td>≥3 ML models + ≥1 NN</td><td>Suggested line-up in §12.</td></tr>
<tr><td>Hyper-parameter tuned per model</td>
    <td>Every suggested model has obvious knobs (C, depth, n_estimators, hidden_layer_sizes, …).</td></tr>
<tr><td>3+ evaluation metrics</td>
    <td>For a 6-class ordinal problem: accuracy, macro-F1, confusion matrix, per-class P/R, MAE on level.</td></tr>
<tr><td>Validation strategy</td><td>Provided test set + stratified 5-fold CV on training (see §13).</td></tr>
<tr><td>Publicly available, cite original</td>
    <td>Warncke-Wang et al. (CHI 2015) and 2017 update.</td></tr>
</tbody></table>
<div class="callout ok"><b>Bottom line:</b> well-sized, structured, ordinal (rare and interesting),
supports both 'structural-only' and 'text-content' research questions.</div>

<h2>3. Filesystem anatomy</h2>
<div class="tree">comp90049-a2/
├── README.md
├── pyproject.toml / uv.lock / .python-version
├── .gitignore
├── Context/                            # assignment brief + reference papers
├── Raw/                                # downloaded dataset (gitignored)
│   ├── 2015_english_wikipedia_quality_dataset/
│   │   ├── datasets/training-set.tsv    (26,506 rows)
│   │   ├── datasets/test-set.tsv        ( 2,941 rows)
│   │   └── revisiondata/                30,273 files · 639 MB
│   └── 2017_english_wikipedia_quality_dataset/
│       ├── datasets/training-set.tsv    (29,174 rows)
│       ├── datasets/test-set.tsv        ( 3,286 rows)
│       └── revisiondata/                33,323 files · 727 MB
├── data/                               # pipeline outputs (gitignored)
│   ├── processed/                      # labels.parquet, features.parquet
│   ├── models/                         # trained model pickles
│   ├── metrics/                        # per-model JSON
│   ├── figures/                        # report PNGs
│   └── reports/                        # eda.html, explainer.html
├── src/a2/
│   ├── config.py · io.py · features.py
│   └── pipeline/
│       ├── prepare.py    # TSV → labels.parquet
│       ├── extract.py    # wikitext → features.parquet
│       ├── train.py      # fit models (stub)
│       ├── evaluate.py   # aggregate metrics
│       ├── report.py     # eda.html
│       └── explainer.py  # explainer.html
├── dashboard.py                        # streamlit app
└── notebooks/                          # per-person exploration</div>
<div class="callout info"><b>Why one file per revid?</b> The dataset is content-addressable:
knowing a revid is enough to fetch its snapshot. Cheap to ship but slow to walk —
never iterate the directory in your pipeline; index by revid instead.</div>

<h2>4. Raw TSV schemas</h2>
<div class="cards">
<div class="card"><h4>2015 TSV <span class="tag">3 columns</span></h4>
<div class="code">{_esc(schema_15)}</div></div>
<div class="card"><h4>2017 TSV <span class="tag">5 columns</span></h4>
<div class="code">{_esc(schema_17)}</div>
<p class="muted">Extra Talk-page columns enable editor-activity features.</p></div>
</div>

<h3>Rows per year × split</h3>{_df_table(pivot)}
<h3>Class counts</h3>{_df_table(class_pivot)}

<h2>5. Wikipedia quality classes</h2>
<p>Wikipedia editors grade articles on an ordinal scale. The classes are defined by
community policy on WikiProject pages.</p>
{_quality_definitions_html()}
<div class="callout">Treat the label as <b>ordinal</b>, not nominal. A model that predicts
"FA" for a Stub is much worse than one predicting "Start" — your metrics should reflect that.</div>

<h2>6. Proposed unified CSV for modelling</h2>
<p>Each row = one revision; columns = label + engineered features. Schema:</p>
<table class="data"><thead><tr><th>Column</th><th>Type</th><th>Source</th></tr></thead><tbody>
<tr><td><span class="inline-code">year, split</span></td><td>category</td><td>dataset / split</td></tr>
<tr><td><span class="inline-code">pageid, revid</span></td><td>int</td><td>TSV columns</td></tr>
<tr><td><span class="inline-code">label</span></td><td>category</td><td>Stub / Start / C / B / GA / FA</td></tr>
<tr><td><span class="inline-code">level</span></td><td>int 0-5</td><td>ordinal map of label</td></tr>
<tr><td><span class="inline-code">chars, tokens, lines, paragraphs, sentences</span></td>
    <td>int</td><td>length features</td></tr>
<tr><td><span class="inline-code">avg_word_len, avg_sent_len, type_token_ratio</span></td>
    <td>float</td><td>lexical features</td></tr>
<tr><td><span class="inline-code">wikilinks, extlinks, templates, images, refs</span></td>
    <td>int</td><td>markup counts</td></tr>
<tr><td><span class="inline-code">headings, max_heading_depth, tables, list_items</span></td>
    <td>int</td><td>structure features</td></tr>
<tr><td><span class="inline-code">bold, italic, categories</span></td><td>int</td>
    <td>style features</td></tr>
<tr><td><span class="inline-code">has_infobox</span></td><td>0/1</td><td>structural binary</td></tr>
</tbody></table>

<h3>Build it like this</h3>
<div class="code">{_esc(csv_code)}</div>

<h3>One row per class — what the CSV looks like</h3>
{_proposed_csv_html(stats)}
<div class="callout info">Dump the result to <span class="inline-code">parquet</span> — faster
reload than CSV, preserves dtypes. After first build the feature-engineering loop never
touches the 1.4 GB of raw files again.</div>

<h2>7. Feature dictionary</h2>
{_feature_cards_html()}
<div class="callout"><b>Layer-2 features (easy adds):</b> TF-IDF unigrams/bigrams over the
lead section; readability scores (Flesch–Kincaid); outgoing-link entropy; edit-history
metadata (number of contributors); pre-trained sentence embeddings.</div>

<h2>8. What the features actually look like</h2>
<p>All plots use a stratified sample of <b>{per_class} training revisions per (year, class)</b>
({len(stats):,} articles total).</p>
<h3>Which features carry the most signal?</h3>
{feat_figs[0] if feat_figs else ''}
<h3>Per-class distributions for the top discriminative features</h3>
{''.join(feat_figs[1:])}
<h3>Class distributions</h3>{fig_dist}
<h3>2015 vs 2017 training proportions</h3>{fig_cmp}

<h2>9. Raw article samples — one per class per year</h2>
{samples_html}

<h2>10. Year overlap — 2015 ↔ 2017</h2>
{fig_trans}
<div class="callout info">The off-diagonal mass tells you Wikipedia improves articles over
time but rarely by more than one class in two years.</div>

<h2>11. Suggested research questions</h2>
<div class="cards">{rq_cards}</div>

<h2>12. Suggested model line-up</h2>
<table class="data"><thead><tr><th>Model</th><th>Family</th><th>Notes &amp; tunable hp</th></tr></thead>
<tbody>{model_rows}</tbody></table>
<div class="callout">Use the <em>same feature matrix</em> across all models in an RQ; use the
<em>same validation protocol</em>. Differences then reflect inductive bias, not engineering.</div>

<h2>13. Validation strategy</h2>
<ol>
<li><b>Provided test sets are untouched final evaluators.</b> No peeking during model selection.</li>
<li><b>Stratified 5-fold CV on training</b> for hyper-parameter tuning. RandomizedSearchCV is enough.</li>
<li>Report <b>CV mean ± std</b> (training) and <b>test numbers</b> (one shot).</li>
<li><b>Cross-year transfer</b> (optional RQ): train on 2015, evaluate on 2017's labels for shared pageids.</li>
<li><b>Metrics:</b> accuracy, macro-F1, weighted-F1, MAE on level, 6×6 confusion matrix.</li>
</ol>

<h2>14. Risks &amp; gotchas</h2>
<ul>
<li><b>Length is too good a feature.</b> A model using only <span class="inline-code">chars</span>
gets respectable accuracy — fine as baseline, not enough for a 25-mark report.</li>
<li><b>Stub vs Start blurs.</b> Both are 'incomplete' — expect most off-diagonal confusion there.</li>
<li><b>2015 and 2017 share six classes</b> but Wikipedia's editorial bar shifted. Don't naïvely
concatenate them unless justified.</li>
<li><b>The wikitext is messy.</b> Regex counts are robust; full text features (TF-IDF, BERT) need cleanup.</li>
<li><b>Talk-page features (2017 only) are tempting but uneven.</b> If used, report results
with and without.</li>
<li><b>A2 forbids AI-generated paragraphs in the report.</b> Allowed: ideation and
short paraphrasing only. Declare in the AI usage section.</li>
</ul>

<p class="muted">Generated by <span class="inline-code">a2 explainer</span> · re-run any time:
<span class="inline-code">uv run a2 explainer --sample 300</span></p>
</body></html>"""


def main(args: argparse.Namespace | None = None) -> None:
    ensure_dirs()
    per_class = args.sample if args and args.sample else 200
    out = Path(args.out) if args and args.out else REPORTS / "explainer.html"

    t0 = time.time()
    master = load_master()
    train = master[master["split"] == "training"].reset_index(drop=True)
    stats = sample_features(train, per_class)
    _, mat, same, shared = _transition_heatmap(master)
    out.write_text(build_html(master, stats, mat, same, shared, per_class),
                   encoding="utf-8")
    print(f"explainer: wrote {out.relative_to(out.parents[2])} "
          f"({out.stat().st_size/1024:.0f} KB) in {time.time()-t0:.1f}s")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sample", type=int, default=200)
    p.add_argument("--out", default=None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    add_args(ap)
    main(ap.parse_args())
