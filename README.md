# COMP90049 Assignment 2 — Wikipedia Article Quality

Group project for COMP90049: predict the quality grade of an English Wikipedia
article (Stub / Start / C / B / GA / FA) from its raw wikitext, using the
2015 and 2017 quality datasets.

The full assignment brief is in [`Context/assignment.md`](Context/assignment.md).
Reference papers and a dataset card live in the same folder.

---

## Quick start

```bash
# 1. Get the data (not committed). Drop both year folders into Raw/.
#    Final layout: Raw/2015_english_wikipedia_quality_dataset/{datasets,revisiondata}
#                  Raw/2017_english_wikipedia_quality_dataset/{datasets,revisiondata}

# 2. Set up Python env (uv handles everything from pyproject.toml + uv.lock)
uv sync

# 3. Run the example pipeline
uv run a2 list                            # list pipelines and their stages
uv run a2 sample prepare                  # TSVs → data/processed/labels.parquet
uv run a2 sample extract --sample 100     # quick dev features parquet
uv run a2 sample report                   # → data/reports/eda.html
uv run a2 sample explainer                # → data/reports/explainer.html

# 4. Interactive dashboard (optional, slower first load)
uv run streamlit run dashboard.py
```

You should now have:
- `data/processed/labels.parquet` — 61,907 rows, normalized across both years
- `data/processed/features-sample100.parquet` — engineered features (sample)
- `data/reports/eda.html` — fast static EDA
- `data/reports/explainer.html` — long-form explainer tied to the A2 rubric

---

## Repo layout

```
comp90049-a2/
├── README.md                              # you are here
├── pyproject.toml + uv.lock               # uv project
├── .python-version                        # 3.12
├── .gitignore                             # ignores Raw/ and data/
│
├── Context/                               # assignment + reference reading
│   ├── assignment.md
│   ├── dataset.md
│   └── paper1.md / paper2.md
│
├── Raw/                                   # downloaded dataset (gitignored)
│   ├── 2015_english_wikipedia_quality_dataset/
│   │   ├── datasets/{training,test}-set.tsv
│   │   └── revisiondata/<revid>           # one raw-wikitext file per revid
│   └── 2017_english_wikipedia_quality_dataset/…
│
├── data/                                  # pipeline outputs (gitignored)
│   ├── processed/                         # labels.parquet, features.parquet
│   ├── models/                            # trained model pickles
│   ├── metrics/                           # per-model JSON
│   ├── figures/                           # PNGs for the report
│   └── reports/                           # eda.html, explainer.html
│
├── src/a2/                                # the importable package
│   ├── config.py                          # paths, class order, feature keys
│   ├── io.py                              # TSV loader, revision reader
│   ├── features.py                        # 21 markup/structural features
│   ├── cli.py                             # `a2 <pipeline> <stage>` dispatcher
│   └── pipelines/
│       ├── README.md
│       └── sample/                        # ← worked example, copy to spin up your own
│           ├── prepare.py · extract.py · train.py · evaluate.py
│           └── report.py · explainer.py
│
├── dashboard.py                           # streamlit app
└── notebooks/                             # per-person sandboxes
```

---

## Pipelines model

Every experiment lives in its own pipeline directory under
`src/a2/pipelines/<name>/`. A pipeline is just a folder of stage modules;
the CLI auto-discovers them.

```bash
uv run a2 list                            # what pipelines exist, what stages each has
uv run a2 sample prepare                  # run one stage
uv run a2 sample all                      # run every stage in order (prepare → … → explainer)
```

The `sample` pipeline is the worked example. Copy it to spin up your own:

```bash
cp -r src/a2/pipelines/sample src/a2/pipelines/<your-name>
# edit prepare.py / extract.py / train.py to taste
uv run a2 <your-name> train --model rf
```

### Stage contract

Every stage module exposes:

```python
def add_args(parser: argparse.ArgumentParser) -> None: ...
def main(args: argparse.Namespace) -> None: ...
```

Stage names are conventional (`prepare`, `extract`, `train`, `evaluate`,
`report`, `explainer`) but not enforced — name them whatever helps the
experiment. Stages whose names match the conventional list run in that
order when you call `a2 <pipeline> all`.

### Where outputs go

| Stage      | Reads                              | Writes                              |
|------------|------------------------------------|-------------------------------------|
| prepare    | `Raw/`                             | `data/processed/labels.parquet`     |
| extract    | `labels.parquet`, `Raw/`           | `data/processed/features.parquet`   |
| train      | `features.parquet`                 | `data/models/<rq>-<model>.joblib`, `data/metrics/<rq>-<model>.json` |
| evaluate   | `data/metrics/*.json`              | `data/figures/*.png`, stdout table  |
| report     | `Raw/`                             | `data/reports/eda.html`             |
| explainer  | `Raw/`                             | `data/reports/explainer.html`       |

Everything in `data/` is regenerable — never commit it. Each pipeline can
override paths via flags (`extract --sample 200`, `train --features-path …`)
so two people can run concurrently without trampling each other's outputs.

---

## Working as a team

The intent: each person owns one or more pipelines + a notebook folder, with
shared code in `src/a2/{config,io,features}.py`.

### Suggested workflow

1. **Pull `main`, then branch.** Branch name = your name or your RQ:
   `git checkout -b <name>/rq-structural`.
2. **Copy the `sample` pipeline** as your starting point:
   `cp -r src/a2/pipelines/sample src/a2/pipelines/<your-name>`.
3. **Modify your pipeline freely.** It only affects your branch. Shared
   modules (`config.py`, `io.py`, `features.py`) only change via PR.
4. **Notebooks** go in `notebooks/<your-name>/`. They import from `a2.*`
   so logic stays in the package, not the notebook.
5. **Open PRs early and small.** Reviewer = the other teammate.

### When you both want to change shared code

- Adding a new feature → add it in `src/a2/features.py`, append the key to
  `FEATURE_KEYS` in `src/a2/config.py`. PR with a one-line description of
  what the feature measures.
- Changing a path or constant → `src/a2/config.py`. PR; usually trivial.
- Anything that changes `labels.parquet` schema → coordinate; both
  pipelines depend on it.

### Commit conventions

Short imperative subject + optional body. Group related changes into one
commit, not 5 micro-commits. Examples:

```
feat(features): add Flesch–Kincaid readability score
fix(prepare): normalise FA -> FA in 2017 (was 'FA ')
pipeline(nathan): switch RQ1 baseline to GBM
```

---

## A2 deliverables checklist

| Deliverable           | Due (5 PM) | Where               |
|-----------------------|------------|---------------------|
| Group contract        | Mon May 4  | Canvas PDF          |
| Report (PDF, ACL)     | Fri May 22 | Canvas              |
| Code (zip)            | Fri May 22 | Canvas              |
| Group reflection      | Fri May 29 | Feedback Fruits     |

Per the brief:
- 3-person team: ≥2 research questions, ≥3 ML models + ≥1 NN
- 4-person team: ≥3 RQs, ≥5 models including NNs
- ≥1 pre-processing/feature-construction step
- ≥3 evaluation metrics
- ≥1 table, ≥1 figure in the report
- Cite the original dataset paper (Warncke-Wang et al.)
- Declare any AI usage in a separate section

For research-question and model ideas tied to this dataset, regenerate
`data/reports/explainer.html` and read sections 11–13.

---

## Troubleshooting

- **`prepare` fails with `FileNotFoundError`** — `Raw/` is empty. Download
  the two dataset folders into `Raw/` first.
- **Streamlit hangs on first load** — it's watching `Raw/` (63k files).
  Run with `--server.fileWatcherType=none` or just use the static reports
  (`a2 sample report` / `a2 sample explainer`).
- **Out-of-memory during `extract`** — extract reads one wikitext file at a
  time; OOM is unlikely. If it happens, pass `--year 2015 --split training`
  to chunk it.
- **`a2: command not found`** — `uv sync` again. The `a2` script is
  registered by `pyproject.toml` and only appears after install.

---

## Data provenance

Original datasets:
- 2015: Warncke-Wang, Cosley, & Riedl (2013). *Tell me more: an actionable
  quality model for Wikipedia.* WikiSym.
- 2017: Updated release with talk-page revids.

Both are publicly available via the WikiClass project. Each row in the
TSVs points to a Wikipedia revision; the raw wikitext for that revision
lives in `Raw/<year>_*/revisiondata/<revid>` as a plain-text file.
