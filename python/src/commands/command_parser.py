"""
Command line argument parsing for training and inference.
"""

import argparse
import os
import sys
import time


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

    if command == "webcrawler":
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
