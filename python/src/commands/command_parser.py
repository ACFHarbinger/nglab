"""
Command line argument parsing for training and inference.
"""

import argparse
import os
import sys
import time
from loguru import logger


class ConfigsParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser with error handling.
    """

    def error(self, message):
        """Handle parsing errors."""
        print(message, end=" ")
        self.print_help()
        sys.exit(2)


from python.src.commands.registry import get_parser


def process_arguments():
    """
    Parse and process command line arguments.

    Returns:
        tuple: (command, arguments_dict)
    """
    parser = get_parser()
    command, args = parser.parse_process_args()

    if command == "active-learning":
        logger.info(
            f"Running Active Learning selection with method: {args.get('method')}"
        )
        sys.exit(0)

    if command == "sentiment":
        from python.src.pipeline.sentiment.analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        if args.get("crawl"):
            from python.src.pipeline.sentiment.news_crawler import main_crawler

            news = main_crawler()
            results = analyzer.analyze([item["title"] for item in news[:5]])
            for res in results:
                sentiment_str = str(res["sentiment"])
                print(f"[{sentiment_str.upper()}] {res['text']}")
        elif args.get("text"):
            text_to_analyze = args.get("text")
            if text_to_analyze:
                res = analyzer.analyze(text_to_analyze)[0]
                sentiment_str = str(res["sentiment"])
                print(f"[{sentiment_str.upper()}] {res['text']}")
        sys.exit(0)

    if command == "train":
        args["run_name"] = "{}_{}".format(
            args["run_name"], time.strftime("%Y%m%dT%H%M%S")
        )
        args["save_dir"] = os.path.join(
            args["output_dir"],
            "{}_{}".format(args.get("data_dir", "default"), args["model"]),
            args["run_name"],
        )
        return command, args

    return command, args
