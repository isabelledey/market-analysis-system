from __future__ import annotations

import importlib
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import main as root_main
import stock_pattern_model
import stock_pattern_model.cli as cli_module
from stock_pattern_model.analysis import analyze_dataframe
from stock_pattern_model.cli import ExitCode
from stock_pattern_model.cli import NO_MATCHING_STOCK_MESSAGE
from stock_pattern_model.cli import _timeframe_menu_text
from stock_pattern_model.cli import main
from stock_pattern_model.config import TIMEFRAME_TO_PERIOD_INTERVAL
from stock_pattern_model.domain import ResolvedInstrument
from stock_pattern_model.exceptions import MarketDataError
from stock_pattern_model.exceptions import UnknownSecurityNumberError
from stock_pattern_model.resolver import CsvInstrumentResolver
from stock_pattern_model.resolver import is_numeric_security_number
from stock_pattern_model.resolver import normalize_identifier


EXCHANGE_TZ = ZoneInfo("America/New_York")

# Captured before the autouse fixture below can monkeypatch the module attribute, so tests
# that specifically want the *real* default validator can still reach it.
_REAL_DEFAULT_INSTRUMENT_VALIDATOR = cli_module._default_instrument_validator


@pytest.fixture(autouse=True)
def _stub_default_instrument_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must never hit the network by default. Tests that specifically exercise the
    resolve+validate retry loop pass their own `validator=` to `main()` to override this."""
    monkeypatch.setattr(cli_module, "_default_instrument_validator", lambda instrument: None)


def make_cli_df(length: int = 30, start: str = "2026-07-10 09:30") -> pd.DataFrame:
    datetimes = pd.date_range(start=start, periods=length, freq="15min", tz=EXCHANGE_TZ)
    rows = []

    for timestamp in datetimes:
        rows.append(
            {
                "Datetime": timestamp,
                "Open": 100.00,
                "High": 100.60,
                "Low": 99.60,
                "Close": 100.10,
                "Volume": 1000,
            }
        )

    rows[-2]["Open"] = 101.00
    rows[-2]["High"] = 101.20
    rows[-2]["Low"] = 99.80
    rows[-2]["Close"] = 100.00
    rows[-2]["Volume"] = 3000
    rows[-1]["Open"] = 99.90
    rows[-1]["High"] = 101.50
    rows[-1]["Low"] = 99.70
    rows[-1]["Close"] = 101.30
    rows[-1]["Volume"] = 3200
    return pd.DataFrame(rows)


def offline_analyzer(symbol: str, **kwargs) -> dict:
    instrument = kwargs.get("instrument") or ResolvedInstrument(
        input_identifier=symbol,
        symbol=symbol,
        name=symbol,
        exchange="Unknown",
        currency="Unknown",
        exchange_timezone="America/New_York",
    )
    as_of = kwargs.get("as_of") or pd.Timestamp("2026-07-10 17:01", tz=EXCHANGE_TZ)
    return analyze_dataframe(
        df=make_cli_df(),
        symbol=symbol,
        interval=kwargs.get("interval", "15m"),
        as_of=as_of,
        display_timezone=kwargs.get("display_timezone", "Asia/Jerusalem"),
        lookback_bars=kwargs.get("lookback_bars", 12),
        top_pattern_count=kwargs.get("top_pattern_count", 3),
        instrument=instrument,
    )


def test_package_imports() -> None:
    assert callable(stock_pattern_model.analyze_stock)
    assert callable(stock_pattern_model.analyze_dataframe)


def test_importing_main_has_no_side_effects(capsys: pytest.CaptureFixture[str]) -> None:
    importlib.reload(root_main)
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""


def test_importing_package_has_no_side_effects(capsys: pytest.CaptureFixture[str]) -> None:
    importlib.reload(stock_pattern_model)
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""


def test_ticker_normalization() -> None:
    assert normalize_identifier("  aapl  ") == "AAPL"


def test_teva_ta_preservation() -> None:
    resolver = CsvInstrumentResolver()
    resolved = resolver.resolve(" teva.ta ")

    assert resolved.symbol == "TEVA.TA"
    assert resolved.exchange == "TASE"


def test_safe_metadata_fallback_for_normal_ticker() -> None:
    resolver = CsvInstrumentResolver()
    resolved = resolver.resolve("aapl")

    assert resolved.symbol == "AAPL"
    assert resolved.exchange == "Unknown"
    assert resolved.currency == "Unknown"


def test_numeric_security_number_detection() -> None:
    assert is_numeric_security_number("1084128") is True
    assert is_numeric_security_number("AAPL") is False


def test_successful_csv_mapping(tmp_path: Path) -> None:
    mapping_file = tmp_path / "tase.csv"
    mapping_file.write_text(
        "security_number,yahoo_symbol,name,exchange,currency,timezone\n"
        "1084128,TEVA.TA,Teva Pharmaceutical Industries,TASE,ILS,Asia/Jerusalem\n",
        encoding="utf-8",
    )
    resolver = CsvInstrumentResolver()

    resolved = resolver.resolve("1084128", mapping_file=str(mapping_file))

    assert resolved.symbol == "TEVA.TA"
    assert resolved.security_number == "1084128"


def test_unknown_security_number_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mapping_file = tmp_path / "tase.csv"
    mapping_file.write_text(
        "security_number,yahoo_symbol,name,exchange,currency,timezone\n"
        "1084128,TEVA.TA,Teva Pharmaceutical Industries,TASE,ILS,Asia/Jerusalem\n",
        encoding="utf-8",
    )

    exit_code = main(
        ["analyze", "9999999", "--mapping-file", str(mapping_file)],
        analyzer=offline_analyzer,
    )
    captured = capsys.readouterr()

    # Unknown security numbers are now unified under the same "no matching stock" error and
    # exit code as any other unresolved ticker/company name (see the interactive validation
    # loop) instead of a dedicated exit code, so CLI and interactive callers see one consistent
    # failure mode.
    assert exit_code == ExitCode.INVALID_INPUT
    assert NO_MATCHING_STOCK_MESSAGE in captured.out


def test_missing_mapping_file_error() -> None:
    exit_code = main(
        ["analyze", "1084128", "--mapping-file", "missing.csv"],
        analyzer=offline_analyzer,
    )

    assert exit_code == ExitCode.MISSING_MAPPING_FILE


def test_invalid_mapping_file_error(tmp_path: Path) -> None:
    mapping_file = tmp_path / "bad.csv"
    mapping_file.write_text(
        "security_number,yahoo_symbol,name\n"
        "1084128,TEVA.TA,Teva Pharmaceutical Industries\n",
        encoding="utf-8",
    )

    exit_code = main(
        ["analyze", "1084128", "--mapping-file", str(mapping_file)],
        analyzer=offline_analyzer,
    )

    assert exit_code == ExitCode.INVALID_MAPPING_FILE


def test_interactive_input_behavior(capsys: pytest.CaptureFixture[str]) -> None:
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return " aapl "

    exit_code = main(
        ["analyze", "--format", "json"],
        input_fn=fake_input,
        analyzer=offline_analyzer,
        interactive=False,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.SUCCESS
    assert prompts == ["Enter a ticker or Israeli security number: "]
    assert '"symbol": "AAPL"' in captured.out


def test_empty_interactive_input_reprompts_instead_of_exiting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Empty input must not crash or exit -- it must be treated the same as any other
    # unresolved identifier and re-prompt until something valid is entered.
    responses = iter(["   ", "", "AAPL"])

    def fake_input(prompt: str) -> str:
        return next(responses)

    exit_code = main(
        ["analyze", "--format", "json"],
        input_fn=fake_input,
        analyzer=offline_analyzer,
        interactive=False,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.SUCCESS
    assert captured.out.count(NO_MATCHING_STOCK_MESSAGE) == 2
    assert '"symbol": "AAPL"' in captured.out


# ---------------------------------------------------------------------------
# Interactive ticker/company-name/security-number resolution and validation
# ---------------------------------------------------------------------------


def test_interactive_valid_ticker_resolves_on_first_attempt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    validated: list[str] = []

    def accepting_validator(instrument: ResolvedInstrument) -> None:
        validated.append(instrument.symbol)

    exit_code = main(
        ["analyze", "--format", "json"],
        input_fn=lambda prompt: "AAPL",
        analyzer=offline_analyzer,
        interactive=False,
        validator=accepting_validator,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.SUCCESS
    assert validated == ["AAPL"]
    assert NO_MATCHING_STOCK_MESSAGE not in captured.out


def test_interactive_unresolvable_ticker_then_valid_ticker_reprompts_until_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Simulates a ticker/company name that does not correspond to any real security: the
    # provider-backed validator rejects it, the user is shown the unified error, and the loop
    # keeps prompting -- it must not fall through to the timeframe menu on the bad attempt.
    responses = iter(["NOTAREALCOMPANY", "AAPL"])

    def fake_input(prompt: str) -> str:
        return next(responses)

    def validator(instrument: ResolvedInstrument) -> None:
        if instrument.symbol == "NOTAREALCOMPANY":
            raise MarketDataError(f"No data returned for symbol '{instrument.symbol}'.")

    exit_code = main(
        ["analyze", "--format", "json"],
        input_fn=fake_input,
        analyzer=offline_analyzer,
        interactive=False,
        validator=validator,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.SUCCESS
    assert captured.out.count(NO_MATCHING_STOCK_MESSAGE) == 1
    assert '"symbol": "AAPL"' in captured.out


def test_interactive_provider_failure_reprompts_until_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The provider itself can fail transiently (network error, rate limit, etc.) -- this must
    # be handled the same way as an unresolvable ticker: show the message and keep prompting,
    # never crash with an unhandled exception.
    attempts = {"count": 0}

    def flaky_validator(instrument: ResolvedInstrument) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise MarketDataError("Failed to load market data for AAPL from Yahoo Finance.")

    exit_code = main(
        ["analyze", "--format", "json"],
        input_fn=lambda prompt: "AAPL",
        analyzer=offline_analyzer,
        interactive=False,
        validator=flaky_validator,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.SUCCESS
    assert attempts["count"] == 2
    assert captured.out.count(NO_MATCHING_STOCK_MESSAGE) == 1


def test_interactive_never_reaches_timeframe_menu_on_invalid_ticker(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Requirement: do not continue to the timeframe menu until the security is valid. Only
    # one timeframe-menu response ("4") is supplied; if the invalid identifier attempt
    # incorrectly fell through to the timeframe menu, that response would be consumed at the
    # wrong point and this would fail with a StopIteration or a mismatched period/interval.
    responses = iter(["BADCOMPANY", "AAPL", "4"])

    def fake_input(prompt: str) -> str:
        return next(responses)

    def validator(instrument: ResolvedInstrument) -> None:
        if instrument.symbol == "BADCOMPANY":
            raise MarketDataError("No data returned.")

    captured_kwargs: dict[str, object] = {}

    def capturing_analyzer(symbol: str, **kwargs) -> dict:
        captured_kwargs.update(kwargs)
        return {"symbol": symbol}

    exit_code = main(
        ["analyze", "--format", "json"],
        input_fn=fake_input,
        analyzer=capturing_analyzer,
        interactive=True,
        validator=validator,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.SUCCESS
    assert captured.out.count(NO_MATCHING_STOCK_MESSAGE) == 1
    assert captured.out.count("Selected timeframe: Three months (3_MONTHS)") == 1
    assert (captured_kwargs["period"], captured_kwargs["interval"]) == ("3mo", "1d")


def test_interactive_israeli_security_number_still_resolves(tmp_path: Path) -> None:
    mapping_file = tmp_path / "tase.csv"
    mapping_file.write_text(
        "security_number,yahoo_symbol,name,exchange,currency,timezone\n"
        "1084128,TEVA.TA,Teva Pharmaceutical Industries,TASE,ILS,Asia/Jerusalem\n",
        encoding="utf-8",
    )
    captured_kwargs: dict[str, object] = {}

    def capturing_analyzer(symbol: str, **kwargs) -> dict:
        captured_kwargs.update(kwargs)
        return {"symbol": symbol}

    exit_code = main(
        ["analyze", "--format", "json", "--mapping-file", str(mapping_file)],
        input_fn=lambda prompt: "1084128",
        analyzer=capturing_analyzer,
        interactive=False,
    )

    assert exit_code == ExitCode.SUCCESS
    assert captured_kwargs["instrument"].symbol == "TEVA.TA"


def test_invalid_cli_ticker_exits_nonzero_without_prompting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_input(prompt: str) -> str:
        raise AssertionError("input_fn should not be called for an explicit CLI ticker")

    def rejecting_validator(instrument: ResolvedInstrument) -> None:
        raise MarketDataError(f"No data returned for symbol '{instrument.symbol}'.")

    exit_code = main(
        ["analyze", "NOTAREALTICKER"],
        input_fn=unexpected_input,
        analyzer=offline_analyzer,
        validator=rejecting_validator,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.INVALID_INPUT
    assert NO_MATCHING_STOCK_MESSAGE in captured.out


def test_invalid_cli_security_number_exits_nonzero_without_prompting(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    mapping_file = tmp_path / "tase.csv"
    mapping_file.write_text(
        "security_number,yahoo_symbol,name,exchange,currency,timezone\n"
        "1084128,TEVA.TA,Teva Pharmaceutical Industries,TASE,ILS,Asia/Jerusalem\n",
        encoding="utf-8",
    )

    def unexpected_input(prompt: str) -> str:
        raise AssertionError("input_fn should not be called for an explicit CLI security number")

    exit_code = main(
        ["analyze", "9999999", "--mapping-file", str(mapping_file)],
        input_fn=unexpected_input,
        analyzer=offline_analyzer,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.INVALID_INPUT
    assert NO_MATCHING_STOCK_MESSAGE in captured.out


def test_default_instrument_validator_rejects_symbol_with_no_market_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Requirement: do not accept a symbol just because the provider returned an empty
    # object -- the real default validator must propagate the provider's own rejection of an
    # empty/nonexistent result rather than swallowing it or accepting the symbol regardless.
    class _EmptyResultProvider:
        def load(self, *, symbol, **kwargs):
            raise MarketDataError(f"No data returned for symbol '{symbol}'.")

    monkeypatch.setattr(cli_module, "YFinanceProvider", lambda *args, **kwargs: _EmptyResultProvider())

    instrument = ResolvedInstrument(
        input_identifier="FAKE",
        symbol="FAKE",
        name="FAKE",
        exchange="Unknown",
        currency="Unknown",
        exchange_timezone=None,
    )

    with pytest.raises(MarketDataError):
        _REAL_DEFAULT_INSTRUMENT_VALIDATOR(instrument)


def test_cli_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        ["analyze", "AAPL", "--format", "json"], analyzer=offline_analyzer, interactive=False
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.SUCCESS
    assert '"symbol": "AAPL"' in captured.out
    assert '"all_detected_patterns"' in captured.out


def test_cli_text_output(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["analyze", "AAPL"], analyzer=offline_analyzer, interactive=False)
    captured = capsys.readouterr()

    assert exit_code == ExitCode.SUCCESS
    assert "Instrument: AAPL" in captured.out
    assert "Resolved Symbol: AAPL" in captured.out
    assert "Exchange Timezone:" in captured.out
    assert "Display Timezone:" in captured.out
    assert "Volume Score:" in captured.out
    assert "Pattern Start:" in captured.out
    assert "Detected at:" in captured.out
    assert "Family:" in captured.out
    assert "Display Detected at:" not in captured.out
    assert "EDT" not in captured.out
    assert "EST" not in captured.out
    assert "Latest Completed Candle Start:" in captured.out
    assert "Asia/Jerusalem" in captured.out
    assert "Detected at:" in captured.out


def test_output_file_creation(tmp_path: Path) -> None:
    output_file = tmp_path / "result.json"

    exit_code = main(
        ["analyze", "AAPL", "--format", "json", "--output", str(output_file)],
        analyzer=offline_analyzer,
        interactive=False,
    )

    assert exit_code == ExitCode.SUCCESS
    assert output_file.exists()
    assert '"symbol": "AAPL"' in output_file.read_text(encoding="utf-8")


def test_cli_forwards_session_mode_to_analyzer() -> None:
    captured_kwargs: dict[str, object] = {}

    def capturing_analyzer(symbol: str, **kwargs) -> dict:
        captured_kwargs.update(kwargs)
        return offline_analyzer(symbol, **kwargs)

    exit_code = main(
        ["analyze", "AAPL", "--session-mode", "regular-and-afterhours", "--format", "json"],
        analyzer=capturing_analyzer,
        interactive=False,
    )

    assert exit_code == ExitCode.SUCCESS
    assert captured_kwargs["session_mode"] == "regular-and-afterhours"


def test_invalid_interval_rejection() -> None:
    with pytest.raises(SystemExit) as error:
        main(["analyze", "AAPL", "--interval", "2hours"], analyzer=offline_analyzer)

    assert error.value.code == 2


@pytest.mark.parametrize("timeframe,expected", sorted(TIMEFRAME_TO_PERIOD_INTERVAL.items()))
def test_timeframe_maps_to_expected_period_and_interval(timeframe: str, expected: tuple[str, str]) -> None:
    captured_kwargs: dict[str, object] = {}

    def capturing_analyzer(symbol: str, **kwargs) -> dict:
        captured_kwargs.update(kwargs)
        return {"symbol": symbol}

    exit_code = main(
        ["analyze", "AAPL", "--timeframe", timeframe, "--format", "json"],
        analyzer=capturing_analyzer,
    )

    assert exit_code == ExitCode.SUCCESS
    assert (captured_kwargs["period"], captured_kwargs["interval"]) == expected


def test_timeframe_combined_with_period_raises_configuration_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["analyze", "AAPL", "--timeframe", "1_MONTH", "--period", "3mo"],
        analyzer=offline_analyzer,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.INVALID_INPUT
    assert "--timeframe cannot be combined with --period or --interval" in captured.out


def test_timeframe_combined_with_interval_raises_configuration_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["analyze", "AAPL", "--timeframe", "1_MONTH", "--interval", "1d"],
        analyzer=offline_analyzer,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.INVALID_INPUT
    assert "--timeframe cannot be combined with --period or --interval" in captured.out


def test_invalid_timeframe_rejection() -> None:
    with pytest.raises(SystemExit) as error:
        main(["analyze", "AAPL", "--timeframe", "2_WEEKS"], analyzer=offline_analyzer)

    assert error.value.code == 2


def test_timeframe_combines_with_as_of_for_look_back_testing() -> None:
    captured_kwargs: dict[str, object] = {}

    def capturing_analyzer(symbol: str, **kwargs) -> dict:
        captured_kwargs.update(kwargs)
        return {"symbol": symbol}

    exit_code = main(
        [
            "analyze",
            "AAPL",
            "--timeframe",
            "6_MONTHS",
            "--as-of",
            "2026-07-10T16:46:00-04:00",
            "--format",
            "json",
        ],
        analyzer=capturing_analyzer,
    )

    assert exit_code == ExitCode.SUCCESS
    assert (captured_kwargs["period"], captured_kwargs["interval"]) == ("6mo", "1d")
    assert captured_kwargs["as_of"] == pd.Timestamp("2026-07-10T16:46:00-04:00")


def test_no_timeframe_keeps_existing_default_period_and_interval() -> None:
    captured_kwargs: dict[str, object] = {}

    def capturing_analyzer(symbol: str, **kwargs) -> dict:
        captured_kwargs.update(kwargs)
        return offline_analyzer(symbol, **kwargs)

    exit_code = main(
        ["analyze", "AAPL", "--format", "json"],
        analyzer=capturing_analyzer,
        interactive=False,
    )

    assert exit_code == ExitCode.SUCCESS
    assert captured_kwargs["period"] == "1mo"
    assert captured_kwargs["interval"] == "15m"


def test_ticker_flag_is_equivalent_to_positional_identifier() -> None:
    captured_kwargs: dict[str, object] = {}

    def capturing_analyzer(symbol: str, **kwargs) -> dict:
        captured_kwargs.update(kwargs)
        return {"symbol": symbol}

    exit_code = main(
        ["analyze", "--ticker", "PYPL", "--timeframe", "3_MONTHS", "--format", "json"],
        analyzer=capturing_analyzer,
        interactive=False,
    )

    assert exit_code == ExitCode.SUCCESS
    assert captured_kwargs.get("instrument") is not None
    assert (captured_kwargs["period"], captured_kwargs["interval"]) == ("3mo", "1d")


def test_ticker_flag_and_positional_identifier_together_raises_configuration_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["analyze", "PYPL", "--ticker", "AAPL", "--no-interactive"],
        analyzer=offline_analyzer,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.INVALID_INPUT
    assert "either as a positional argument or with --ticker" in captured.out


def test_interactive_run_prompts_for_timeframe_when_omitted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_kwargs: dict[str, object] = {}
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "4"

    def capturing_analyzer(symbol: str, **kwargs) -> dict:
        captured_kwargs.update(kwargs)
        return {"symbol": symbol}

    exit_code = main(
        ["analyze", "AAPL", "--format", "json"],
        input_fn=fake_input,
        analyzer=capturing_analyzer,
        interactive=True,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.SUCCESS
    assert prompts == [_timeframe_menu_text()]
    assert (captured_kwargs["period"], captured_kwargs["interval"]) == ("3mo", "1d")
    assert "Selected timeframe: Three months (3_MONTHS)" in captured.out


def test_interactive_run_retries_on_invalid_timeframe_choice_until_valid(
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses = iter(["next_tuesday", "", "0", "8", "6"])

    def fake_input(prompt: str) -> str:
        return next(responses)

    captured_kwargs: dict[str, object] = {}

    def capturing_analyzer(symbol: str, **kwargs) -> dict:
        captured_kwargs.update(kwargs)
        return {"symbol": symbol}

    exit_code = main(
        ["analyze", "AAPL", "--format", "json"],
        input_fn=fake_input,
        analyzer=capturing_analyzer,
        interactive=True,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.SUCCESS
    assert (captured_kwargs["period"], captured_kwargs["interval"]) == ("1y", "1d")
    assert "Invalid choice: 'next_tuesday'. Please enter a number from 1 to 7." in captured.out
    assert "Invalid choice: ''. Please enter a number from 1 to 7." in captured.out
    assert "Invalid choice: '0'. Please enter a number from 1 to 7." in captured.out
    assert "Invalid choice: '8'. Please enter a number from 1 to 7." in captured.out
    assert "Selected timeframe: One year (1_YEAR)" in captured.out


def test_interactive_run_never_raises_on_invalid_timeframe_input() -> None:
    # Requirement: invalid input must never crash or terminate the program -- it must keep
    # re-prompting until a valid 1-7 selection is made.
    responses = iter(["not-a-number", "5"])

    def fake_input(prompt: str) -> str:
        return next(responses)

    def capturing_analyzer(symbol: str, **kwargs) -> dict:
        return {"symbol": symbol}

    exit_code = main(
        ["analyze", "AAPL", "--format", "json"],
        input_fn=fake_input,
        analyzer=capturing_analyzer,
        interactive=True,
    )

    assert exit_code == ExitCode.SUCCESS


def test_interactive_run_does_not_prompt_when_timeframe_flag_given() -> None:
    captured_kwargs: dict[str, object] = {}

    def capturing_analyzer(symbol: str, **kwargs) -> dict:
        captured_kwargs.update(kwargs)
        return {"symbol": symbol}

    def unexpected_input(prompt: str) -> str:
        raise AssertionError(f"input_fn should not be called, got prompt: {prompt!r}")

    exit_code = main(
        ["analyze", "AAPL", "--timeframe", "1_YEAR", "--format", "json"],
        input_fn=unexpected_input,
        analyzer=capturing_analyzer,
        interactive=True,
    )

    assert exit_code == ExitCode.SUCCESS
    assert (captured_kwargs["period"], captured_kwargs["interval"]) == ("1y", "1d")


def test_interactive_run_does_not_prompt_when_period_given_manually() -> None:
    captured_kwargs: dict[str, object] = {}

    def capturing_analyzer(symbol: str, **kwargs) -> dict:
        captured_kwargs.update(kwargs)
        return {"symbol": symbol}

    def unexpected_input(prompt: str) -> str:
        raise AssertionError(f"input_fn should not be called, got prompt: {prompt!r}")

    exit_code = main(
        ["analyze", "AAPL", "--period", "3mo", "--interval", "1d", "--format", "json"],
        input_fn=unexpected_input,
        analyzer=capturing_analyzer,
        interactive=True,
    )

    assert exit_code == ExitCode.SUCCESS
    assert (captured_kwargs["period"], captured_kwargs["interval"]) == ("3mo", "1d")


def test_non_interactive_run_does_not_prompt_and_uses_default() -> None:
    def unexpected_input(prompt: str) -> str:
        raise AssertionError(f"input_fn should not be called, got prompt: {prompt!r}")

    exit_code = main(
        ["analyze", "AAPL", "--format", "json"],
        input_fn=unexpected_input,
        analyzer=offline_analyzer,
        interactive=False,
    )

    assert exit_code == ExitCode.SUCCESS


def test_cli_interactive_flag_forces_prompt_even_when_stdin_is_not_a_tty() -> None:
    # No `interactive=` kwarg is passed here, so this exercises the real
    # sys.stdin.isatty() fallback path, overridden by --interactive. Under
    # pytest, stdin is never a tty, matching consoles (e.g. PyCharm's default
    # Run window) that don't allocate a pseudo-terminal either.
    captured_kwargs: dict[str, object] = {}
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "2"

    def capturing_analyzer(symbol: str, **kwargs) -> dict:
        captured_kwargs.update(kwargs)
        return {"symbol": symbol}

    exit_code = main(
        ["analyze", "AAPL", "--interactive", "--format", "json"],
        input_fn=fake_input,
        analyzer=capturing_analyzer,
    )

    assert exit_code == ExitCode.SUCCESS
    assert prompts == [_timeframe_menu_text()]
    assert (captured_kwargs["period"], captured_kwargs["interval"]) == ("5d", "15m")


def test_cli_no_interactive_flag_skips_prompt() -> None:
    def unexpected_input(prompt: str) -> str:
        raise AssertionError(f"input_fn should not be called, got prompt: {prompt!r}")

    exit_code = main(
        ["analyze", "AAPL", "--no-interactive", "--format", "json"],
        input_fn=unexpected_input,
        analyzer=offline_analyzer,
    )

    assert exit_code == ExitCode.SUCCESS


def test_keyboard_interrupt_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    def interrupted_analyzer(symbol: str, **kwargs) -> dict:
        raise KeyboardInterrupt

    exit_code = main(["analyze", "AAPL"], analyzer=interrupted_analyzer, interactive=False)
    captured = capsys.readouterr()

    assert exit_code == ExitCode.INTERRUPTED
    assert captured.out.strip() == "Analysis interrupted."


def test_invalid_timezone_rejection(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        ["analyze", "AAPL", "--display-timezone", "Mars/Base"],
        analyzer=offline_analyzer,
        interactive=False,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.INVALID_TIMEZONE
    assert "Unknown display timezone" in captured.out


def test_nonzero_exit_codes_for_invalid_output_target(tmp_path: Path) -> None:
    output_path = tmp_path / "missing-dir" / "result.json"

    exit_code = main(
        ["analyze", "AAPL", "--format", "json", "--output", str(output_path)],
        analyzer=offline_analyzer,
        interactive=False,
    )

    assert exit_code == ExitCode.OUTPUT_FILE_FAILURE


def test_cli_offline_file_analysis(capsys: pytest.CaptureFixture[str]) -> None:
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "sample_ohlcv.csv"

    exit_code = main(
        [
            "analyze",
            "AAPL",
            "--data-file",
            str(fixture_path),
            "--exchange-timezone",
            "America/New_York",
            "--format",
            "json",
        ],
        interactive=False,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.SUCCESS
    assert '"data_quality_report"' in captured.out


def test_missing_data_file_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "analyze",
            "TEST",
            "--data-file",
            "missing.csv",
            "--exchange-timezone",
            "America/New_York",
        ],
        interactive=False,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.MISSING_DATA_FILE
    assert "Data file not found" in captured.out


def test_package_help_output(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    captured = capsys.readouterr()

    assert error.value.code == 0
    assert "analyze" in captured.out
    assert "backtest" not in captured.out


def test_analyze_help_output(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["analyze", "--help"])
    captured = capsys.readouterr()

    assert error.value.code == 0
    assert "--all-patterns" in captured.out


def test_root_main_interactive_delegation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_package_main(argv):
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(root_main, "package_main", fake_package_main)

    exit_code = root_main.main([])

    assert exit_code == 0
    assert calls == [["analyze"]]


def test_root_main_positional_ticker_delegation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_package_main(argv):
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(root_main, "package_main", fake_package_main)

    exit_code = root_main.main(["AAPL"])

    assert exit_code == 0
    assert calls == [["analyze", "AAPL"]]


def test_root_main_analyze_delegation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_package_main(argv):
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(root_main, "package_main", fake_package_main)

    exit_code = root_main.main(["analyze", "TEVA.TA"])

    assert exit_code == 0
    assert calls == [["analyze", "TEVA.TA"]]


def test_root_main_help_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_package_main(argv):
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(root_main, "package_main", fake_package_main)

    exit_code = root_main.main(["--help"])

    assert exit_code == 0
    assert calls == [["--help"]]


def test_root_main_uses_sys_argv_when_no_explicit_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_package_main(argv):
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(root_main, "package_main", fake_package_main)
    monkeypatch.setattr(root_main.sys, "argv", ["main.py", "--help"])

    exit_code = root_main.main()

    assert exit_code == 0
    assert calls == [["--help"]]


def test_root_main_data_file_flag_delegation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_package_main(argv):
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(root_main, "package_main", fake_package_main)

    exit_code = root_main.main(["TEST", "--data-file", "fixture.csv", "--exchange-timezone", "America/New_York"])

    assert exit_code == 0
    assert calls == [["analyze", "TEST", "--data-file", "fixture.csv", "--exchange-timezone", "America/New_York"]]


def test_legacy_root_modules_remain_importable_but_clearly_deprecated() -> None:
    legacy_data_loader = importlib.import_module("data_loader")
    legacy_features = importlib.import_module("features")
    legacy_pattern_detector = importlib.import_module("pattern_detector")
    legacy_model = importlib.import_module("model")

    assert "deprecated compatibility wrapper" in (legacy_data_loader.__doc__ or "").lower()
    assert "deprecated compatibility wrapper" in (legacy_features.__doc__ or "").lower()
    assert "deprecated compatibility wrapper" in (legacy_pattern_detector.__doc__ or "").lower()
    assert "deprecated compatibility wrapper" in (legacy_model.__doc__ or "").lower()
    assert callable(legacy_features.add_features)
    assert callable(legacy_model.analyze_stock)
