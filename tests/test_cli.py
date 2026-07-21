from __future__ import annotations

import importlib
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import main as root_main
import stock_pattern_model
from stock_pattern_model.analysis import analyze_dataframe
from stock_pattern_model.cli import ExitCode
from stock_pattern_model.cli import main
from stock_pattern_model.domain import ResolvedInstrument
from stock_pattern_model.resolver import CsvInstrumentResolver
from stock_pattern_model.resolver import is_numeric_security_number
from stock_pattern_model.resolver import normalize_identifier


EXCHANGE_TZ = ZoneInfo("America/New_York")


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


def test_unknown_security_number_error(tmp_path: Path) -> None:
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

    assert exit_code == ExitCode.UNKNOWN_SECURITY_NUMBER


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

    exit_code = main(["analyze", "--format", "json"], input_fn=fake_input, analyzer=offline_analyzer)
    captured = capsys.readouterr()

    assert exit_code == ExitCode.SUCCESS
    assert prompts == ["Enter a ticker or Israeli security number: "]
    assert '"symbol": "AAPL"' in captured.out


def test_empty_interactive_input_returns_nonzero_exit_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["analyze"], input_fn=lambda prompt: "   ", analyzer=offline_analyzer)
    captured = capsys.readouterr()

    assert exit_code == ExitCode.INVALID_INPUT
    assert "No instrument identifier was provided." in captured.out


def test_cli_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["analyze", "AAPL", "--format", "json"], analyzer=offline_analyzer)
    captured = capsys.readouterr()

    assert exit_code == ExitCode.SUCCESS
    assert '"symbol": "AAPL"' in captured.out
    assert '"all_detected_patterns"' in captured.out


def test_cli_text_output(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["analyze", "AAPL"], analyzer=offline_analyzer)
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
    )

    assert exit_code == ExitCode.SUCCESS
    assert captured_kwargs["session_mode"] == "regular-and-afterhours"


def test_invalid_interval_rejection() -> None:
    with pytest.raises(SystemExit) as error:
        main(["analyze", "AAPL", "--interval", "2hours"], analyzer=offline_analyzer)

    assert error.value.code == 2


def test_keyboard_interrupt_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    def interrupted_analyzer(symbol: str, **kwargs) -> dict:
        raise KeyboardInterrupt

    exit_code = main(["analyze", "AAPL"], analyzer=interrupted_analyzer)
    captured = capsys.readouterr()

    assert exit_code == ExitCode.INTERRUPTED
    assert captured.out.strip() == "Analysis interrupted."


def test_invalid_timezone_rejection(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        ["analyze", "AAPL", "--display-timezone", "Mars/Base"],
        analyzer=offline_analyzer,
    )
    captured = capsys.readouterr()

    assert exit_code == ExitCode.INVALID_TIMEZONE
    assert "Unknown display timezone" in captured.out


def test_nonzero_exit_codes_for_invalid_output_target(tmp_path: Path) -> None:
    output_path = tmp_path / "missing-dir" / "result.json"

    exit_code = main(
        ["analyze", "AAPL", "--format", "json", "--output", str(output_path)],
        analyzer=offline_analyzer,
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
        ]
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
        ]
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
