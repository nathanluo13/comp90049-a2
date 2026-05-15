"""`python -m a2` entry point — just delegates to the CLI."""

from .cli import main

raise SystemExit(main())
