"""Thin compatibility wrapper around the packaged stock-pattern CLI."""

from __future__ import annotations

import sys
from typing import Sequence

from stock_pattern_model.cli import main as package_main


def _normalize_root_argv(argv: Sequence[str] | None) -> list[str]:
    arguments = list(argv or [])
    if not arguments:
        return ["analyze"]
    if arguments[0] in {"analyze", "-h", "--help"}:
        return arguments
    if arguments[0] == "--verbose":
        return ["--verbose", "analyze", *arguments[1:]]
    if arguments[0].startswith("-"):
        return ["analyze", *arguments]
    return ["analyze", *arguments]


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate root-level execution to the packaged CLI."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    return package_main(_normalize_root_argv(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
