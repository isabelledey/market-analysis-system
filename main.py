"""Simple command-line runner for the stock pattern analysis model."""

from __future__ import annotations

from model import analyze_stock


def print_analysis(result: dict) -> None:
    """Print a readable summary for one symbol."""
    print(f"Symbol: {result['symbol']}")
    print(f"Latest Datetime: {result['latest_datetime']}")
    print(f"Latest Close: {result['latest_close']}")
    print(f"Interval: {result['interval']}")
    print(f"Trend: {result['trend']}")
    print(f"Market State: {result['market_state']}")
    print(f"Overall Bias: {result['overall_bias']}")
    print(f"Confidence Score: {result['confidence_score']}")
    print(f"Bullish Score: {result['bullish_score']}")
    print(f"Bearish Score: {result['bearish_score']}")
    print(f"Total Score: {result['total_score']}")
    print(f"Ignored Patterns: {result['ignored_patterns_count']}")
    print("Top Patterns:")

    if result["top_patterns"]:
        for pattern in result["top_patterns"]:
            print(
                f"  - {pattern['pattern']} at {pattern['datetime']} "
                f"({pattern['signal']}, {pattern['candles_ago']} candles ago, "
                f"weighted score {pattern['weighted_score']})"
            )
    else:
        print("  - None in the last 12 candles")

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
