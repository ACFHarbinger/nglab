import argparse
import unittest

from python.src.commands.active_learning_parser import add_active_learning_args
from python.src.commands.crawler_parser import add_crawler_args
from python.src.commands.hpo_parser import add_hpo_args
from python.src.commands.inference_parser import add_inference_args
from python.src.commands.sentiment_parser import add_sentiment_args


class TestCrawlerParser(unittest.TestCase):
    def test_add_crawler_args(self):
        parser = argparse.ArgumentParser()
        add_crawler_args(parser)

        args = parser.parse_args(["--website", "http://example.com"])
        self.assertEqual(args.website, "http://example.com")

    def test_add_crawler_args_url_alias(self):
        parser = argparse.ArgumentParser()
        add_crawler_args(parser)

        args = parser.parse_args(["--url", "http://test.com"])
        self.assertEqual(args.website, "http://test.com")


class TestHPOParser(unittest.TestCase):
    def test_add_hpo_args_defaults(self):
        parser = argparse.ArgumentParser()
        add_hpo_args(parser)

        args = parser.parse_args([])
        self.assertEqual(args.model, "lstm")
        self.assertEqual(args.num_samples, 10)
        self.assertEqual(args.max_epochs, 10)
        self.assertEqual(args.gpus_per_trial, 0.0)
        self.assertEqual(args.seed, 1234)

    def test_add_hpo_args_custom(self):
        parser = argparse.ArgumentParser()
        add_hpo_args(parser)

        args = parser.parse_args(
            [
                "--model",
                "nstransformer",
                "--num_samples",
                "50",
                "--max_epochs",
                "20",
                "--gpus_per_trial",
                "1.0",
                "--data_dir",
                "/data",
                "--seed",
                "42",
            ]
        )

        self.assertEqual(args.model, "nstransformer")
        self.assertEqual(args.num_samples, 50)
        self.assertEqual(args.max_epochs, 20)
        self.assertEqual(args.gpus_per_trial, 1.0)
        self.assertEqual(args.data_dir, "/data")
        self.assertEqual(args.seed, 42)


class TestInferenceParser(unittest.TestCase):
    def test_add_inference_args_defaults(self):
        parser = argparse.ArgumentParser()
        add_inference_args(parser)

        args = parser.parse_args([])
        self.assertEqual(args.model, "lstm")
        self.assertEqual(args.seed, 1234)

    def test_add_inference_args_custom(self):
        parser = argparse.ArgumentParser()
        add_inference_args(parser)

        args = parser.parse_args(
            [
                "--model",
                "nstransformer",
                "--data_dir",
                "/data",
                "--load_path",
                "/models/best.pt",
                "--id",
                "ts_123",
                "--seed",
                "999",
            ]
        )

        self.assertEqual(args.model, "nstransformer")
        self.assertEqual(args.data_dir, "/data")
        self.assertEqual(args.load_path, "/models/best.pt")
        self.assertEqual(args.id, "ts_123")
        self.assertEqual(args.seed, 999)


class TestSentimentParser(unittest.TestCase):
    def test_add_sentiment_args_text(self):
        parser = argparse.ArgumentParser()
        add_sentiment_args(parser)

        args = parser.parse_args(["--text", "This is great news!"])
        self.assertEqual(args.text, "This is great news!")

    def test_add_sentiment_args_file(self):
        parser = argparse.ArgumentParser()
        add_sentiment_args(parser)

        args = parser.parse_args(["--file", "/path/to/file.txt"])
        self.assertEqual(args.file, "/path/to/file.txt")

    def test_add_sentiment_args_crawl(self):
        parser = argparse.ArgumentParser()
        add_sentiment_args(parser)

        args = parser.parse_args(["--crawl"])
        self.assertTrue(args.crawl)


class TestActiveLearningParser(unittest.TestCase):
    def test_add_active_learning_args_defaults(self):
        parser = argparse.ArgumentParser()
        add_active_learning_args(parser)

        args = parser.parse_args([])
        self.assertEqual(args.method, "quantile")
        self.assertEqual(args.budget, 10)

    def test_add_active_learning_args_custom(self):
        parser = argparse.ArgumentParser()
        add_active_learning_args(parser)

        args = parser.parse_args(
            [
                "--method",
                "mc_dropout",
                "--budget",
                "50",
                "--pool_path",
                "/data/pool.csv",
            ]
        )

        self.assertEqual(args.method, "mc_dropout")
        self.assertEqual(args.budget, 50)
        self.assertEqual(args.pool_path, "/data/pool.csv")

    def test_add_active_learning_args_method_choices(self):
        parser = argparse.ArgumentParser()
        add_active_learning_args(parser)

        # Test all valid choices
        for method in ["quantile", "mc_dropout", "random"]:
            args = parser.parse_args(["--method", method])
            self.assertEqual(args.method, method)
