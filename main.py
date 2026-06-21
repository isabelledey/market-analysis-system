"""Simple command-line runner for the stock pattern analysis model."""

from __future__ import annotations

from model import analyze_stock


def print_analysis(result: dict) -> None:
    """Print a readable summary for one symbol."""
    print(f"Symbol: {result['symbol']}")
    print(f"Latest Date: {result['latest_date']}")
    print(f"Latest Close: {result['latest_close']}")
    print(f"Trend: {result['trend']}")
    print(f"Overall Bias: {result['overall_bias']}")
    print(f"Confidence Score: {result['confidence_score']}")
    print("Detected Patterns:")

    if result["detected_patterns"]:
        for pattern in result["detected_patterns"]:
            print(
                f"  - {pattern['pattern']} on {pattern['date']} "
                f"({pattern['signal']}, weighted score {pattern['weighted_score']})"
            )
    else:
        print("  - None in the last 5 trading days")

    print("Explanation:")
    print(f"  {result['explanation']}")
    print("-" * 80)


def main() -> None:
    """Run the analysis model for a small sample watchlist."""
    symbols = ["AAPL", "MSFT", "NVDA", "TSLA"]

    for symbol in symbols:
        try:
            result = analyze_stock(symbol)
            print_analysis(result)
        except Exception as error:
            print(f"Symbol: {symbol}")
            print(f"Error: {error}")
            print("-" * 80)


if __name__ == "__main__":
    main()
