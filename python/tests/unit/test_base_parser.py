import unittest
from unittest.mock import patch

from python.src.commands.base_parser import (
    ConfigsParser,
    LowercaseAction,
    StoreDictKeyPair,
)


class TestConfigsParser(unittest.TestCase):
    def test_error_prints_help(self):
        parser = ConfigsParser()
        with patch("builtins.print"), patch.object(parser, "print_help"):
            with self.assertRaises(SystemExit):
                parser.error("Test error message")

    def test_parse_process_args_with_command(self):
        parser = ConfigsParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        cmd1 = subparsers.add_parser("cmd1")
        cmd1.add_argument("--opt", default="val")

        command, opts = parser.parse_process_args(["cmd1", "--opt", "test"])
        self.assertEqual(command, "cmd1")
        self.assertEqual(opts["opt"], "test")

    def test_parse_process_args_no_command(self):
        parser = ConfigsParser()
        parser.add_subparsers(dest="command")

        with patch("builtins.print"), patch.object(parser, "print_help"):
            with self.assertRaises(SystemExit):
                parser.parse_process_args([])


class TestLowercaseAction(unittest.TestCase):
    def test_lowercase_action(self):
        parser = ConfigsParser()
        parser.add_argument("--name", action=LowercaseAction)

        args = parser.parse_args(["--name", "TEST"])
        self.assertEqual(args.name, "test")

    def test_lowercase_action_none(self):
        parser = ConfigsParser()
        parser.add_argument("--name", action=LowercaseAction, default=None)

        args = parser.parse_args([])
        self.assertIsNone(args.name)


class TestStoreDictKeyPair(unittest.TestCase):
    def test_store_dict_key_pair(self):
        parser = ConfigsParser()
        parser.add_argument("--params", nargs="*", action=StoreDictKeyPair, default={})

        args = parser.parse_args(["--params", "key1=val1", "key2=val2"])
        self.assertEqual(args.params, {"key1": "val1", "key2": "val2"})

    def test_store_dict_key_pair_invalid(self):
        parser = ConfigsParser()
        parser.add_argument("--params", nargs="*", action=StoreDictKeyPair)

        with self.assertRaises(SystemExit):
            parser.parse_args(["--params", "invalidformat"])
