"""
Centralized registry for the modular CLI parser.
"""

from python.src.commands.active_learning_parser import add_active_learning_args
from python.src.commands.backtest_parser import add_backtest_args
from python.src.commands.base_parser import ConfigsParser
from python.src.commands.crawler_parser import add_crawler_args
from python.src.commands.evaluate_parser import add_evaluate_args
from python.src.commands.hpo_parser import add_hpo_args
from python.src.commands.inference_parser import add_inference_args
from python.src.commands.sentiment_parser import add_sentiment_args
from python.src.commands.train_parser import add_train_args


def get_parser() -> ConfigsParser:
    """
    Creates and returns the main ConfigsParser with all subcommands registered.
    """
    parser = ConfigsParser(
        description="Nothing Gambles Like A Bot, your personal stock market assistant!"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="The command to execute", required=True
    )

    # Training
    train_parser = subparsers.add_parser(
        "train", help="Generic training for neural model"
    )
    add_train_args(train_parser)

    # Inference
    inference_parser = subparsers.add_parser("inference", help="Run model inference")
    add_inference_args(inference_parser)

    # Web Crawler
    crawler_parser = subparsers.add_parser(
        "webcrawler", aliases=["crawler"], help="Crawl websites for data"
    )
    add_crawler_args(crawler_parser)

    # HPO (Hyperparameter Optimization)
    hpo_parser = subparsers.add_parser(
        "hp_optim", aliases=["hpo"], help="Hyperparameter optimization"
    )
    add_hpo_args(hpo_parser)

    # Active Learning
    al_parser = subparsers.add_parser(
        "active-learning", aliases=["al"], help="Select informative samples"
    )
    add_active_learning_args(al_parser)

    # Sentiment Analysis
    sentiment_parser = subparsers.add_parser(
        "sentiment", help="Analyze market sentiment"
    )
    add_sentiment_args(sentiment_parser)

    # Evaluation
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate model performance"
    )
    add_evaluate_args(evaluate_parser)

    # Backtesting
    backtest_parser = subparsers.add_parser(
        "backtest", help="Run historical backtest"
    )
    add_backtest_args(backtest_parser)

    return parser
