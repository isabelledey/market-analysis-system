"""Argparse-based command-line interface for stock pattern analysis."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from enum import IntEnum
from pathlib import Path
from typing import Callable

import pandas as pd

from stock_pattern_model.analysis import analyze_stock
from stock_pattern_model.config import (
    SUPPORTED_INTERVALS,
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_TO_PERIOD_INTERVAL,
)
from stock_pattern_model.domain import ResolvedInstrument
from stock_pattern_model.exceptions import (
    ConfigurationError,
    DataValidationError,
    InvalidInstrumentError,
    InvalidMappingFileError,
    MarketDataError,
    MissingDataFileError,
    MissingMappingFileError,
    NoCompletedBarsError,
    OutputFileError,
    UnknownSecurityNumberError,
)
from stock_pattern_model.formatters import format_analysis_json, format_analysis_text
from stock_pattern_model.market_data import YFinanceProvider
from stock_pattern_model.resolver import CsvInstrumentResolver
from stock_pattern_model.session_utils import SUPPORTED_SESSION_MODES

LOGGER = logging.getLogger(__name__)

DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "15m"

NO_MATCHING_STOCK_MESSAGE = (
    "Error: No matching stock was found. Please enter the ticker, company name, "
    "or security number again."
)

# UnknownSecurityNumberError is an InvalidInstrumentError subclass and
# MarketDataProviderError is a MarketDataError subclass, so this tuple already covers both.
_INVALID_INSTRUMENT_ERRORS = (InvalidInstrumentError, MarketDataError)


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
    INTERRUPTED = 130


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


_TIMEFRAME_MENU: tuple[tuple[str, str, str], ...] = (
    ("1", "1_DAY", "One day"),
    ("2", "1_WEEK", "One week"),
    ("3", "1_MONTH", "One month"),
    ("4", "3_MONTHS", "Three months"),
    ("5", "6_MONTHS", "Six months"),
    ("6", "1_YEAR", "One year"),
    ("7", "5_YEARS", "Five years"),
)
_TIMEFRAME_MENU_BY_NUMBER = {number: (value, label) for number, value, label in _TIMEFRAME_MENU}


def _timeframe_menu_text() -> str:
    lines = ["Choose a timeframe:", ""]
    lines.extend(f"{number}) {label}" for number, _value, label in _TIMEFRAME_MENU)
    lines.extend(["", "Enter your choice (1-7): "])
    return "\n".join(lines)


def _prompt_for_timeframe(input_fn: Callable[[str], str]) -> str:
    menu_text = _timeframe_menu_text()
    while True:
        choice = input_fn(menu_text).strip()
        match = _TIMEFRAME_MENU_BY_NUMBER.get(choice)
        if match is not None:
            value, label = match
            print(f"Selected timeframe: {label} ({value})")
            return value
        print(f"Invalid choice: '{choice}'. Please enter a number from 1 to 7.\n")


def _resolve_period_and_interval(
    args: argparse.Namespace,
    input_fn: Callable[[str], str],
    interactive: bool,
) -> tuple[str, str]:
    if args.timeframe is not None:
        if args.period is not None or args.interval is not None:
            raise ConfigurationError(
                "--timeframe cannot be combined with --period or --interval. "
                "Choose --timeframe alone, or set --period/--interval manually."
            )
        return TIMEFRAME_TO_PERIOD_INTERVAL[args.timeframe]
    if args.period is not None or args.interval is not None:
        return args.period or DEFAULT_PERIOD, args.interval or DEFAULT_INTERVAL
    if interactive:
        timeframe = _prompt_for_timeframe(input_fn)
        return TIMEFRAME_TO_PERIOD_INTERVAL[timeframe]
    return DEFAULT_PERIOD, DEFAULT_INTERVAL


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(prog="python -m stock_pattern_model")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    subparsers = parser.add_subparsers(dest="command")
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a ticker or security number.")
    analyze_parser.add_argument("identifier", nargs="?", help="Ticker or Israeli security number.")
    analyze_parser.add_argument(
        "--ticker",
        default=None,
        help="Ticker or Israeli security number (alternative to the positional identifier).",
    )
    analyze_parser.add_argument("--interval", default=None, choices=SUPPORTED_INTERVALS)
    analyze_parser.add_argument("--period", default=None)
    analyze_parser.add_argument(
        "--timeframe",
        choices=SUPPORTED_TIMEFRAMES,
        default=None,
        help=(
            "Preset analysis time range. Automatically selects the matching period "
            "and interval (1_DAY/1_WEEK -> 15m, 1_MONTH -> 1h, 3_MONTHS/6_MONTHS/1_YEAR "
            "-> 1d, 5_YEARS -> 1wk). Cannot be combined with --period or --interval. "
            "Combine with --as-of to look back to a past evening for testing."
        ),
    )
    analyze_parser.add_argument("--lookback-bars", type=int, default=12)
    analyze_parser.add_argument("--top", type=int, default=3)
    analyze_parser.add_argument("--all-patterns", action="store_true")
    analyze_parser.add_argument(
        "--pattern-history",
        choices=("none", "current", "session", "all"),
        default="session",
    )
    analyze_parser.add_argument("--history-limit", type=int)
    analyze_parser.add_argument("--display-timezone", default="Asia/Jerusalem")
    analyze_parser.add_argument(
        "--session-mode",
        choices=SUPPORTED_SESSION_MODES,
        default="regular",
        help="Session segment selection for analysis (default: regular).",
    )
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
    analyze_parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "The timeframe prompt is on by default whenever --timeframe/--period/"
            "--interval are all omitted. Pass --no-interactive to suppress it and "
            "keep the 1mo/15m default instead, for scripts and scheduled jobs."
        ),
    )
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


def _default_instrument_validator(instrument: ResolvedInstrument) -> None:
    """Confirm a resolved instrument refers to a real security with actual price data.

    Resolver-level resolution is just string/CSV normalization -- it never checks whether the
    symbol really exists. This performs a small live fetch and lets the provider's own
    validation (which already rejects an empty result) surface as an error; it does not
    accept a symbol merely because resolution produced *an* object.
    """
    provider = YFinanceProvider()
    provider.load(
        symbol=instrument.symbol,
        interval="1d",
        period="5d",
        exchange_timezone=instrument.exchange_timezone,
    )


def _resolve_and_validate_instrument(
    identifier_arg: str | None,
    input_fn: Callable[[str], str],
    resolver: CsvInstrumentResolver,
    validator: Callable[[ResolvedInstrument], None],
    mapping_file: str | None,
) -> tuple[str, ResolvedInstrument]:
    """Resolve a ticker/company-name/security-number identifier and confirm it refers to a
    real, existing security with actual price data.

    If ``identifier_arg`` is provided (CLI/non-interactive), one failed attempt raises
    immediately with a unified error and no retry. If it is None (interactive), the user is
    re-prompted until a valid, existing security is entered -- this never crashes and never
    silently proceeds without a genuine match.
    """
    if identifier_arg is not None:
        if not identifier_arg.strip():
            raise InvalidInstrumentError(NO_MATCHING_STOCK_MESSAGE)
        try:
            instrument = resolver.resolve(identifier_arg, mapping_file=mapping_file)
            validator(instrument)
        except _INVALID_INSTRUMENT_ERRORS as error:
            raise InvalidInstrumentError(NO_MATCHING_STOCK_MESSAGE) from error
        return identifier_arg, instrument

    while True:
        identifier = input_fn("Enter a ticker or Israeli security number: ")
        if not identifier.strip():
            print(NO_MATCHING_STOCK_MESSAGE)
            continue
        try:
            instrument = resolver.resolve(identifier, mapping_file=mapping_file)
            validator(instrument)
        except _INVALID_INSTRUMENT_ERRORS:
            print(NO_MATCHING_STOCK_MESSAGE)
            continue
        return identifier, instrument


def _run_analyze(
    args: argparse.Namespace,
    input_fn: Callable[[str], str],
    resolver: CsvInstrumentResolver,
    analyzer: Callable[..., dict],
    interactive: bool,
    validator: Callable[[ResolvedInstrument], None],
) -> str:
    if args.identifier is not None and args.ticker is not None:
        raise ConfigurationError(
            "Provide the ticker either as a positional argument or with --ticker, not both."
        )
    if args.identifier is None and args.ticker is not None:
        args.identifier = args.ticker
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
        identifier, instrument = _resolve_and_validate_instrument(
            args.identifier, input_fn, resolver, validator, args.mapping_file
        )
    _validate_positive_int(args.lookback_bars, "--lookback-bars")
    _validate_positive_int(args.top, "--top")
    if args.history_limit is not None:
        _validate_positive_int(args.history_limit, "--history-limit")
    if args.cache_ttl < 0:
        raise ConfigurationError("--cache-ttl must be >= 0.")
    if args.data_file and args.mapping_file and identifier.isdigit():
        LOGGER.debug("Numeric identifier will be resolved through the mapping file.")
    period, interval = _resolve_period_and_interval(args, input_fn=input_fn, interactive=interactive)
    as_of = _parse_as_of(args.as_of)
    LOGGER.debug("Resolved instrument: %s", instrument.to_dict())
    result = analyzer(
        instrument.symbol,
        period=period,
        interval=interval,
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
        session_mode=args.session_mode,
    )

    if args.format == "json":
        return format_analysis_json(result)
    return format_analysis_text(
        result,
        include_all_patterns=args.all_patterns,
        pattern_history_mode=args.pattern_history,
        history_limit=args.history_limit,
    )


def main(
    argv: Sequence[str] | None = None,
    input_fn: Callable[[str], str] = input,
    resolver: CsvInstrumentResolver | None = None,
    analyzer: Callable[..., dict] | None = None,
    interactive: bool | None = None,
    validator: Callable[[ResolvedInstrument], None] | None = None,
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
    validator = validator or _default_instrument_validator
    if interactive is None:
        cli_interactive = getattr(args, "interactive", None)
        interactive = cli_interactive if cli_interactive is not None else True

    try:
        content = _run_analyze(
            args,
            input_fn=input_fn,
            resolver=resolver,
            analyzer=analyzer,
            interactive=interactive,
            validator=validator,
        )
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
    except KeyboardInterrupt:
        print("Analysis interrupted.")
        return ExitCode.INTERRUPTED
    except Exception as error:
        if getattr(args, "verbose", False):
            LOGGER.exception("Unexpected internal failure")
            print(str(error))
        else:
            LOGGER.error("Unexpected internal failure: %s", error)
            print("Unexpected internal failure. Re-run with --verbose for more detail.")
        return ExitCode.INTERNAL_FAILURE

    return ExitCode.SUCCESS
