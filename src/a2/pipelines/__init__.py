"""Each subpackage here is a pipeline — a self-contained chain of stages.

Adding a new pipeline:
  1. Create src/a2/pipelines/<your-name>/ with an empty __init__.py
  2. Add stage modules (prepare.py, extract.py, train.py, evaluate.py, …)
  3. Each stage module exposes:
        main(args)              — runs the stage
        add_args(parser)        — registers CLI flags
  4. Invoke via:
        uv run a2 <your-name> <stage> [flags]

The `sample` pipeline is the worked example — copy it and modify.
"""
