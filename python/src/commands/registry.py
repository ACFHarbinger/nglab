"""
Centralized registry for the modular CLI parser.
"""

from python.src.commands.base_parser import ConfigsParser
from python.src.commands.train_parser import add_train_args
from python.src.commands.inference_parser import add_inference_args
from python.src.commands.crawler_parser import add_crawler_args
from python.src.commands.hpo_parser import add_hpo_args


def get_parser() -> ConfigsParser:
    """
    Creates and returns the main ConfigsParser with all subcommands registered.
    """
    parser = ConfigsParser(
        description="Nothing Gambles Like A Bot, your personal stock market assistant!"
    )
    subparsers = parser.add_subparsers(dest="command", help="The command to execute", required=True)

    # Training
    train_parser = subparsers.add_parser("train", help="Generic training for neural model")
    add_train_args(train_parser)

    # Inference
    inference_parser = subparsers.add_parser("inference", help="Run model inference")
    add_inference_args(inference_parser)

    # Web Crawler
    crawler_parser = subparsers.add_parser("webcrawler", aliases=["crawler"], help="Crawl websites for data")
    add_crawler_args(crawler_parser)

    # HPO (Hyperparameter Optimization)
    hpo_parser = subparsers.add_parser("hp_optim", aliases=["hpo"], help="Hyperparameter optimization")
    add_hpo_args(hpo_parser)

    return parser
