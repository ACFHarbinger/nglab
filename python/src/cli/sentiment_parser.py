"""
Sentiment analysis command parser.
"""

import argparse


def add_sentiment_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the sentiment command."""
    parser.add_argument("--text", type=str, help="Text to analyze for sentiment.")
    parser.add_argument(
        "--file", type=str, help="Path to a file containing text to analyze."
    )
    parser.add_argument(
        "--crawl", action="store_true", help="Crawl news and analyze their sentiment."
    )
