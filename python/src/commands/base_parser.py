"""
Base parser utilities for the modular CLI.
"""

import argparse
import sys
from typing import Any, Dict, List, Optional, Tuple


class ConfigsParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser with enhanced error handling.
    """

    def error(self, message: str) -> None:
        """Handle parsing errors by printing help and exiting."""
        print(message, end=" ")
        self.print_help()
        sys.exit(2)

    def parse_process_args(self, args: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Parses arguments and returns the command and options dictionary.
        """
        if args is None:
            args = sys.argv[1:]

        parsed_args = vars(self.parse_args(args))
        command = parsed_args.pop("command", None)
        
        if command is None:
            self.error("No command specified.")
            
        return str(command), parsed_args


class LowercaseAction(argparse.Action):
    """Action to convert argument value to lowercase."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Invoke action: lowercase input string."""
        if values is not None:
            values = str(values).lower()
        setattr(namespace, self.dest, values)


class StoreDictKeyPair(argparse.Action):
    """Custom action to parse key=value into a dictionary."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Invoke action: parse key=value strings into dictionary."""
        my_dict = {}
        if values:
            for kv in values:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    my_dict[k] = v
                else:
                    raise argparse.ArgumentError(self, f"Could not parse argument '{kv}' as key=value format")
        setattr(namespace, self.dest, my_dict)