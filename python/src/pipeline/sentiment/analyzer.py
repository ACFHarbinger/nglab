"""
Sentiment Analysis for Financial Text.

Integrates pre-trained NLP models (FinBERT) to extract sentiment signals
from news headlines, reports, and social media.
"""

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentAnalyzer:
    """
    Wrapper for FinBERT to analyze sentiment of financial text.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str | None = None):
        """
        Initialize the analyzer.

        Args:
            model_name: HuggingFace model ID.
            device: Device to run the model on ('cuda', 'cpu').
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

        # FinBERT labels: 0: positive, 1: negative, 2: neutral
        self.labels = ["positive", "negative", "neutral"]

    def analyze(self, text: str | list[str]) -> list[dict[str, Any]]:
        """
        Analyze sentiment of the given text(s).

        Args:
            text: Single string or list of strings.

        Returns:
            List of dictionaries with label and scores.
        """
        if isinstance(text, str):
            text = [text]

        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)

        results = []
        for i in range(len(text)):
            scores = probs[i].cpu().numpy()
            results.append(
                {
                    "text": text[i],
                    "sentiment": self.labels[scores.argmax()],
                    "scores": {
                        "positive": float(scores[0]),
                        "negative": float(scores[1]),
                        "neutral": float(scores[2]),
                    },
                }
            )

        return results
