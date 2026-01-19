"""
Web crawler command parser.
"""

import argparse


def add_crawler_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the webcrawler command."""
    parser.add_argument(
        "--website", "--url", type=str, help="URL of the website to crawl for data."
    )
