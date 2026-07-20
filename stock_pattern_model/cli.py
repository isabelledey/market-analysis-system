"""Argparse-based command-line interface for stock pattern analysis."""

from __future__ import annotations

import argparse
import logging
from enum import IntEnum
from pathlib import Path
from typing import Callable
from typing import Sequence

import pandas as pd

from stock_pattern_model.analysis import analyze_stock
from stock_pattern_model.config import SUPPORTED_INTERVALS
from stock_pattern_model.domain import ResolvedInstrument
from stock_pattern_model.exceptions import ConfigurationError
from stock_pattern_model.exceptions import DataValidationError
from stock_pattern_model.exceptions import InvalidMappingFileError
from stock_pattern_model.exceptions import InvalidInstrumentError
from stock_pattern_model.exceptions import MarketDataError
from stock_pattern_model.exceptions import MissingDataFileError
from stock_pattern_model.exceptions import MissingMappingFileError
from stock_pattern_model.exceptions import NoCompletedBarsError
from stock_pattern_model.exceptions import OutputFileError
from stock_pattern_model.exceptions import UnknownSecurityNumberError
from stock_pattern_model.formatters import format_analysis_json
from stock_pattern_model.formatters import format_analysis_text
from stock_pattern_model.resolver import CsvInstrumentResolver


LOGGER = logging.getLogger(__name__)


class ExitCode(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 2
    UNKNOWN_SECURITY_NUMBER = 3
    MISSING_MAPPING_FILE = 4
    INVALID_MAPPING_FILE = 5
    MARKET_DATA_FAILURE = 6
    DATA_VALIDATION_FAILURE = 7
    OUTPUT_FILE_FAILURE = 8
    INVALID_TIMEZONE = 9
    NO_COMPLETED_BARS = 10
    MISSING_DATA_FILE = 11
    INTERNAL_FAILURE = 12


def _parse_as_of(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise ConfigurationError("--as-of must include timezone information.")
    return parsed


def _validate_positive_int(value: int, flag_name: str) -> None:
    if value < 1:
        raise ConfigurationError(f"{flag_name} must be at least 1.")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(prog="python -m stock_pattern_model")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    subparsers = parser.add_subparsers(dest="command")
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a ticker or security number.")
    analyze_parser.add_argument("identifier", nargs="?", help="Ticker or Israeli security number.")
    analyze_parser.add_argument("--interval", default="15m", choices=SUPPORTED_INTERVALS)
    analyze_parser.add_argument("--period", default="1mo")
    analyze_parser.add_argument("--lookback-bars", type=int, default=12)
    analyze_parser.add_argument("--top", type=int, default=3)
    analyze_parser.add_argument("--all-patterns", action="store_true")
    analyze_parser.add_argument("--display-timezone", default="Asia/Jerusalem")
    analyze_parser.add_argument("--format", choices=("text", "json"), default="text")
    analyze_parser.add_argument("--output")
    analyze_parser.add_argument("--mapping-file")
    analyze_parser.add_argument("--data-file")
    analyze_parser.add_argument("--exchange-timezone")
    analyze_parser.add_argument("--cache-dir")
    analyze_parser.add_argument("--cache-ttl", type=int, default=3600)
    analyze_parser.add_argument("--no-cache", action="store_true")
    analyze_parser.add_argument(
        "--strict-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strict market-data validation (default: enabled).",
    )
    analyze_parser.add_argument("--as-of")
    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", force=True)


def _write_output(content: str, output_path: str | None) -> None:
    if not output_path:
        print(content)
        return

    path = Path(output_path)
    try:
        path.write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")
    except OSError as error:
        raise OutputFileError(f"Could not write output file: {output_path}") from error


def _prompt_for_identifier(input_fn: Callable[[str], str]) -> str:
    identifier = input_fn("Enter a ticker or Israeli security number: ")
    if not identifier.strip():
        raise InvalidInstrumentError("No instrument identifier was provided.")
    return identifier


def _run_analyze(
    args: argparse.Namespace,
    input_fn: Callable[[str], str],
    resolver: CsvInstrumentResolver,
    analyzer: Callable[..., dict],
) -> str:
    if args.identifier is None and args.data_file:
        identifier = Path(args.data_file).stem
        instrument = ResolvedInstrument(
            input_identifier=identifier,
            symbol=identifier.strip().upper(),
            name=identifier.strip().upper(),
            exchange="Offline File",
            currency="Unknown",
            exchange_timezone=args.exchange_timezone,
        )
    else:
        identifier = args.identifier or _prompt_for_identifier(input_fn)
        instrument = resolver.resolve(identifier, mapping_file=args.mapping_file)
    _validate_positive_int(args.lookback_bars, "--lookback-bars")
    _validate_positive_int(args.top, "--top")
    if args.cache_ttl < 0:
        raise ConfigurationError("--cache-ttl must be >= 0.")
    if args.data_file and args.mapping_file and identifier.isdigit():
        LOGGER.debug("Numeric identifier will be resolved through the mapping file.")
    as_of = _parse_as_of(args.as_of)
    LOGGER.debug("Resolved instrument: %s", instrument.to_dict())
    result = analyzer(
        instrument.symbol,
        period=args.period,
        interval=args.interval,
        as_of=as_of,
        lookback_bars=args.lookback_bars,
        top_pattern_count=args.top,
        display_timezone=args.display_timezone,
        instrument=instrument,
        data_file=args.data_file,
        exchange_timezone=args.exchange_timezone,
        cache_dir=args.cache_dir,
        cache_ttl=args.cache_ttl,
        no_cache=args.no_cache,
        strict_data=args.strict_data,
    )

    if args.format == "json":
        return format_analysis_json(result)
    return format_analysis_text(result, include_all_patterns=args.all_patterns)


def main(
    argv: Sequence[str] | None = None,
    input_fn: Callable[[str], str] = input,
    resolver: CsvInstrumentResolver | None = None,
    analyzer: Callable[..., dict] | None = None,
) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))

    if args.command is None:
        parser.print_help()
        return ExitCode.SUCCESS

    resolver = resolver or CsvInstrumentResolver()
    analyzer = analyzer or analyze_stock

    try:
        content = _run_analyze(args, input_fn=input_fn, resolver=resolver, analyzer=analyzer)
        _write_output(content, args.output)
    except UnknownSecurityNumberError as error:
        print(str(error))
        return ExitCode.UNKNOWN_SECURITY_NUMBER
    except MissingMappingFileError as error:
        print(str(error))
        return ExitCode.MISSING_MAPPING_FILE
    except InvalidMappingFileError as error:
        print(str(error))
        return ExitCode.INVALID_MAPPING_FILE
    except InvalidInstrumentError as error:
        print(str(error))
        return ExitCode.INVALID_INPUT
    except ConfigurationError as error:
        print(str(error))
        if "timezone" in str(error).lower():
            return ExitCode.INVALID_TIMEZONE
        return ExitCode.INVALID_INPUT
    except NoCompletedBarsError as error:
        print(str(error))
        return ExitCode.NO_COMPLETED_BARS
    except MissingDataFileError as error:
        print(str(error))
        return ExitCode.MISSING_DATA_FILE
    except MarketDataError as error:
        print(str(error))
        return ExitCode.MARKET_DATA_FAILURE
    except DataValidationError as error:
        print(str(error))
        return ExitCode.DATA_VALIDATION_FAILURE
    except OutputFileError as error:
        print(str(error))
        return ExitCode.OUTPUT_FILE_FAILURE
    except Exception as error:  # noqa: BLE001
        if getattr(args, "verbose", False):
            LOGGER.exception("Unexpected internal failure")
            print(str(error))
        else:
            LOGGER.error("Unexpected internal failure: %s", error)
            print("Unexpected internal failure. Re-run with --verbose for more detail.")
        return ExitCode.INTERNAL_FAILURE

    return ExitCode.SUCCESS
