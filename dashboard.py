"""Streamlit dashboard — interactive EDA over the Wikipedia quality datasets.

Run with:
    uv run streamlit run dashboard.py
"""

from __future__ import annotations

import random

import pandas as pd
import plotly.express as px
import streamlit as st

from a2.config import CLASS_COLORS, CLASS_ORDER, DATASETS
from a2.features import features
from a2.io import load_master, load_split, read_revision


# ---------- cached loaders ---------- #

@st.cache_data(show_spinner=False)
def _master() -> pd.DataFrame:
    return load_master()


@st.cache_data(show_spinner=False)
def _split(year: str, split: str) -> pd.DataFrame:
    return load_split(year, split)


@st.cache_data(show_spinner=False, max_entries=512)
def _revision(year: str, revid: int) -> str | None:
    return read_revision(year, revid)


@st.cache_data(show_spinner="Sampling revisions for stats…")
def _sample_stats(year: str, split: str, per_class: int, seed: int = 42) -> pd.DataFrame:
    df = _split(year, split)
    rng = random.Random(seed)
    rows = []
    for label, group in df.groupby("label"):
        idxs = list(group.index)
        rng.shuffle(idxs)
        for idx in idxs[:per_class]:
            r = df.loc[idx]
            text = _revision(year, int(r["revid"]))
            if not text:
                continue
            f = features(text)
            f.update({"label": label, "pageid": int(r["pageid"]),
                      "revid": int(r["revid"]), "year": year, "split": split})
            rows.append(f)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["label"] = pd.Categorical(out["label"], categories=CLASS_ORDER, ordered=True)
    return out


# ---------- UI ---------- #

st.set_page_config(page_title="A2 Wikipedia Quality — EDA", layout="wide")

st.sidebar.title("A2 Dashboard")
page = st.sidebar.radio(
    "Page",
    ["Overview", "Class Distributions", "Article Length & Markup",
     "Raw Article Viewer", "Year Overlap"],
)

master = _master()

with st.sidebar.expander("Dataset summary", expanded=True):
    for year, cfg in DATASETS.items():
        n_rev = sum(1 for _ in (cfg["dir"] / "revisiondata").iterdir())
        st.write(f"**{year}** — {n_rev:,} revisions on disk")
    st.write(f"Total label rows: **{len(master):,}**")


# Overview --------------------------------------------------------- #
if page == "Overview":
    st.title("COMP90049 A2 — Wikipedia Quality Datasets")

    cols = st.columns(4)
    cols[0].metric("Years", len(DATASETS))
    cols[1].metric("Label rows", f"{len(master):,}")
    cols[2].metric("Train rows", f"{(master['split']=='training').sum():,}")
    cols[3].metric("Test rows", f"{(master['split']=='test').sum():,}")

    st.subheader("Rows by year and split")
    pivot = (master.groupby(["year", "split"]).size()
             .unstack(fill_value=0))
    pivot["total"] = pivot.sum(axis=1)
    st.dataframe(pivot, use_container_width=True)

    st.subheader("Schema preview")
    for year in DATASETS:
        with st.expander(f"{year} — TSV head + columns"):
            df = _split(year, "training")
            st.code(", ".join(df.columns), language="text")
            st.dataframe(df.head(8), use_container_width=True)

    st.subheader("Raw revision file — example")
    year = st.selectbox("Year", list(DATASETS), key="ov_year")
    df = _split(year, "training")
    sample = df.sample(1, random_state=1).iloc[0]
    st.write(f"pageid `{sample['pageid']}` · revid `{sample['revid']}` · label **{sample['label']}**")
    text = _revision(year, int(sample["revid"])) or "(missing)"
    st.code(text[:2000] + ("…" if len(text) > 2000 else ""), language="markdown")


# Class Distributions --------------------------------------------- #
elif page == "Class Distributions":
    st.title("Class Distributions")
    proportion = st.toggle("Show proportions instead of counts", value=False)
    for year in DATASETS:
        st.subheader(year)
        df = master[master["year"] == year]
        counts = df.groupby(["split", "label"]).size().reset_index(name="n")
        counts["label"] = pd.Categorical(counts["label"], categories=CLASS_ORDER, ordered=True)
        if proportion:
            counts["n"] = counts.groupby("split")["n"].transform(lambda s: s / s.sum())
            yaxis = "n"
        else:
            yaxis = "n"
        fig = px.bar(counts, x="label", y=yaxis, color="label", facet_col="split",
                     category_orders={"label": CLASS_ORDER},
                     color_discrete_map=CLASS_COLORS,
                     text=counts[yaxis].apply(lambda v: f"{v:.1%}" if proportion else f"{int(v)}"))
        fig.update_layout(showlegend=False, height=380, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)


# Length & Markup ------------------------------------------------- #
elif page == "Article Length & Markup":
    st.title("Article Length & Markup")
    col1, col2, col3 = st.columns(3)
    year = col1.selectbox("Year", list(DATASETS), key="len_year")
    split = col2.selectbox("Split", ["training", "test"], key="len_split")
    per_class = col3.slider("Revisions per class", 25, 500, 100, step=25)
    stats = _sample_stats(year, split, per_class)
    st.subheader("Summary stats by class")
    summary = (stats.groupby("label", observed=True)
               [["chars", "tokens", "wikilinks", "templates", "headings", "refs", "images"]]
               .median().round(0).astype(int))
    st.dataframe(summary, use_container_width=True)
    metric = st.selectbox("Metric",
                          ["chars", "tokens", "wikilinks", "templates",
                           "headings", "refs", "images"])
    log = st.toggle("Log y-axis", value=metric in ("chars", "tokens"))
    fig = px.box(stats, x="label", y=metric, color="label",
                 category_orders={"label": CLASS_ORDER},
                 color_discrete_map=CLASS_COLORS, points="suspectedoutliers")
    if log:
        fig.update_yaxes(type="log")
    fig.update_layout(showlegend=False, height=460, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


# Raw Article Viewer ---------------------------------------------- #
elif page == "Raw Article Viewer":
    st.title("Raw Article Viewer")
    col1, col2, col3, col4 = st.columns(4)
    year = col1.selectbox("Year", list(DATASETS), key="rv_year")
    split = col2.selectbox("Split", ["training", "test"], key="rv_split")
    df = _split(year, split)
    labels = ["(all)"] + [c for c in CLASS_ORDER if c in df["label"].unique()]
    label = col3.selectbox("Class", labels, key="rv_label")
    if label != "(all)":
        df = df[df["label"] == label]
    seed = col4.number_input("Seed", value=0, step=1)
    n_show = st.slider("How many to list", 1, 50, 10)
    sample = df.sample(min(n_show, len(df)), random_state=int(seed))
    st.write(f"{len(df):,} rows match · showing {len(sample)}")
    for _, row in sample.iterrows():
        with st.expander(f"[{row['label']}] pageid={row['pageid']} · revid={row['revid']}"):
            text = _revision(year, int(row["revid"]))
            if text is None:
                st.error("Revision file not found.")
                continue
            stats = features(text)
            cols = st.columns(7)
            for c, (k, v) in zip(cols, list(stats.items())[:7]):
                c.metric(k, f"{v:,}")
            st.code(text, language="markdown")


# Year Overlap ---------------------------------------------------- #
elif page == "Year Overlap":
    st.title("Year Overlap — 2015 vs 2017")
    p15 = set(master[master["year"] == "2015"]["pageid"])
    p17 = set(master[master["year"] == "2017"]["pageid"])
    inter = p15 & p17
    cols = st.columns(3)
    cols[0].metric("Pages in 2015", f"{len(p15):,}")
    cols[1].metric("Pages in 2017", f"{len(p17):,}")
    cols[2].metric("Shared", f"{len(inter):,}")
    if not inter:
        st.info("No shared pageids.")
        st.stop()
    a = master[(master["year"] == "2015") & (master["pageid"].isin(inter))][
        ["pageid", "label"]].rename(columns={"label": "label_2015"}).drop_duplicates("pageid")
    b = master[(master["year"] == "2017") & (master["pageid"].isin(inter))][
        ["pageid", "label"]].rename(columns={"label": "label_2017"}).drop_duplicates("pageid")
    joined = a.merge(b, on="pageid")
    mat = (joined.groupby(["label_2015", "label_2017"]).size().unstack(fill_value=0)
           .reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0))
    st.dataframe(mat, use_container_width=True)
    fig = px.imshow(mat, text_auto=True, aspect="auto",
                    labels=dict(x="2017 label", y="2015 label", color="pages"),
                    color_continuous_scale="Blues")
    fig.update_layout(height=520, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    same = (joined["label_2015"] == joined["label_2017"]).sum()
    st.metric("Unchanged label", f"{same:,} / {len(joined):,}",
              delta=f"{same/len(joined):.1%}")
