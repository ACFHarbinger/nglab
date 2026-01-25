import argparse
import unittest
from unittest.mock import patch

from python.src.commands.registry import get_parser
from python.src.commands.train_parser import add_train_args


class TestRegistry(unittest.TestCase):
    @patch("python.src.commands.registry.add_train_args")
    @patch("python.src.commands.registry.add_inference_args")
    @patch("python.src.commands.registry.add_crawler_args")
    @patch("python.src.commands.registry.add_hpo_args")
    @patch("python.src.commands.registry.add_active_learning_args")
    @patch("python.src.commands.registry.add_sentiment_args")
    def test_get_parser(self, *mocks):
        """Test get_parser creates parser and calls all add_*_args functions."""
        parser = get_parser()
        self.assertIsNotNone(parser)

        # Verify all add_*_args functions were called
        for mock in mocks:
            mock.assert_called_once()


class TestTrainParser(unittest.TestCase):
    def test_add_train_args(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)

        # Parse with defaults
        args = parser.parse_args([])
        self.assertEqual(args.model, "lstm")
        self.assertEqual(args.n_seq, 1)
        self.assertEqual(args.seq_len, 21)
        self.assertEqual(args.pred_len, 3)
        self.assertEqual(args.embedding_dim, 128)
        self.assertEqual(args.hidden_dim, 128)
        self.assertEqual(args.n_encode_layers, 2)
        self.assertEqual(args.n_epochs, 100)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.lr_model, 1e-4)
        self.assertEqual(args.lr_decay, 1.0)
        self.assertEqual(args.seed, 1234)
        self.assertFalse(args.no_cuda)
        self.assertEqual(args.max_grad_norm, 1.0)
        self.assertEqual(args.log_dir, "logs")
        self.assertEqual(args.run_name, "run")
        self.assertEqual(args.output_dir, "results")
        self.assertIsNone(args.load_path)
        self.assertEqual(args.epoch_start, 0)
        self.assertFalse(args.no_tensorboard)
        self.assertFalse(args.no_progress_bar)
        self.assertEqual(args.checkpoint_epochs, 100)
        self.assertEqual(args.log_step, 1)
        self.assertFalse(args.distributed)
        self.assertEqual(args.local_rank, 0)

    def test_add_train_args_custom(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)

        # Parse with custom values
        args = parser.parse_args(
            [
                "--model",
                "nstransformer",
                "--n_epochs",
                "50",
                "--batch_size",
                "32",
                "--lr_model",
                "0.001",
                "--no_cuda",
                "--distributed",
            ]
        )

        self.assertEqual(args.model, "nstransformer")
        self.assertEqual(args.n_epochs, 50)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.lr_model, 0.001)
        self.assertTrue(args.no_cuda)
        self.assertTrue(args.distributed)
