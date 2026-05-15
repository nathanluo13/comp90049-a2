"""Top-level CLI — `a2 <pipeline> <stage> [flags]`.

Pipelines are auto-discovered from `src/a2/pipelines/<name>/`. Each stage is a
module exposing `main(args)` and optionally `add_args(parser)`.

Listing what's available:
    uv run a2 list

Invoking a stage:
    uv run a2 sample prepare
    uv run a2 sample extract --sample 200

Running every stage of a pipeline (best-effort, in module-name order, skips
'all' itself):
    uv run a2 sample all
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from types import ModuleType

from . import pipelines as pipelines_pkg


# Stages run in this order when invoking `all`. Stages not listed here are
# skipped from `all` (e.g. one-off experiments).
STAGE_ORDER = ["prepare", "extract", "train", "evaluate", "report", "explainer"]


def discover_pipelines() -> dict[str, ModuleType]:
    out: dict[str, ModuleType] = {}
    for info in pkgutil.iter_modules(pipelines_pkg.__path__):
        if info.ispkg:
            out[info.name] = importlib.import_module(
                f"{pipelines_pkg.__name__}.{info.name}"
            )
    return dict(sorted(out.items()))


def discover_stages(pipeline_name: str) -> dict[str, ModuleType]:
    pkg = importlib.import_module(f"{pipelines_pkg.__name__}.{pipeline_name}")
    out: dict[str, ModuleType] = {}
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.ispkg or info.name.startswith("_"):
            continue
        out[info.name] = importlib.import_module(
            f"{pipelines_pkg.__name__}.{pipeline_name}.{info.name}"
        )
    return dict(sorted(out.items()))


def cmd_list(_argv: list[str]) -> int:
    pipelines = discover_pipelines()
    if not pipelines:
        print("(no pipelines found in src/a2/pipelines/)")
        return 0
    print("Pipelines:")
    for name in pipelines:
        stages = list(discover_stages(name))
        print(f"  {name:<14}  stages: {', '.join(stages)}")
    return 0


def _run_stage(pipeline: str, stage: str, argv: list[str]) -> int:
    stages = discover_stages(pipeline)
    if stage not in stages:
        print(f"unknown stage '{stage}' in pipeline '{pipeline}'. "
              f"Available: {', '.join(stages)}", file=sys.stderr)
        return 2
    mod = stages[stage]
    parser = argparse.ArgumentParser(prog=f"a2 {pipeline} {stage}")
    if hasattr(mod, "add_args"):
        mod.add_args(parser)
    ns = parser.parse_args(argv)
    mod.main(ns)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        cmd_list([])
        return 0
    if argv[0] == "list":
        return cmd_list(argv[1:])

    pipelines = discover_pipelines()
    pipeline = argv[0]
    if pipeline not in pipelines:
        print(f"unknown pipeline '{pipeline}'. Available: {', '.join(pipelines)}",
              file=sys.stderr)
        return 2

    if len(argv) < 2:
        print(f"usage: a2 {pipeline} <stage> [flags]   (or `all`)", file=sys.stderr)
        cmd_list([])
        return 2

    stage = argv[1]
    rest = argv[2:]

    if stage == "all":
        stages = discover_stages(pipeline)
        for s in STAGE_ORDER:
            if s in stages:
                print(f"\n=== {pipeline}.{s} ===")
                rc = _run_stage(pipeline, s, [])
                if rc:
                    return rc
        return 0

    return _run_stage(pipeline, stage, rest)


if __name__ == "__main__":
    raise SystemExit(main())
