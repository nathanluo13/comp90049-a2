# Pipelines

Each subdirectory here is a self-contained pipeline. The `sample/` pipeline is
the worked example.

## Add your own

```bash
cp -r src/a2/pipelines/sample src/a2/pipelines/<your-name>
# edit the stage modules; rename them; add new ones; whatever
uv run a2 <your-name> <stage>
```

## Stage contract

Each stage module is a Python file (no extension prefix) that exposes:

```python
def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--foo", default=1)

def main(args: argparse.Namespace) -> None:
    # do the work
    ...
```

The CLI imports the module and calls `add_args` (if present) then `main`.

## Conventional stage names

| Name      | Role                                                       |
|-----------|------------------------------------------------------------|
| prepare   | TSVs → canonical labels parquet                            |
| extract   | Raw wikitext → features parquet                            |
| train     | Fit a model; write to `data/models/` and `data/metrics/`   |
| evaluate  | Aggregate metrics; produce figures for the paper           |
| report    | Static EDA HTML                                            |
| explainer | Long-form HTML walkthrough                                 |

Only these run when you call `a2 <pipeline> all`. Any other stage names
(e.g. `tune`, `ensemble`, `error_analysis`) are first-class — they just
have to be invoked explicitly.

## Why pipelines per person

Shared `data/processed/labels.parquet` and `src/a2/features.py` mean both
people generate the same canonical inputs. Pipeline directories let each
person run their own train/evaluate/report stages without stepping on each
other. Output files inside `data/models/` and `data/metrics/` are tagged
by `--rq` / `--model`, so concurrent runs do not collide.
