"""Naive Bayes model suite."""

from typing import Any

from sklearn.naive_bayes import GaussianNB, MultinomialNB

from ..base import ClassicalModel


class NaiveBayesModel(ClassicalModel):
    """
    Naive Bayes wrapper supporting Gaussian and Multinomial variants.
    """

    def __init__(self, nb_type: str = "gaussian", **kwargs: Any) -> None:
        """
        Initialize the Naive Bayes model.

        Args:
            nb_type (str, optional): 'gaussian' or 'multinomial'. Defaults to "gaussian".
            **kwargs: Additional arguments passed to the underlying sklearn model.
        """
        super().__init__()
        if nb_type == "gaussian":
            self.model = GaussianNB(**kwargs)
        else:
            self.model = MultinomialNB(**kwargs)


class GaussianNaiveBayesModel(NaiveBayesModel):
    """Gaussian Naive Bayes classifier."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Gaussian Naive Bayes model."""
        super().__init__(nb_type="gaussian", **kwargs)


class MultinomialNaiveBayesModel(NaiveBayesModel):
    """Multinomial Naive Bayes classifier."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Multinomial Naive Bayes model."""
        super().__init__(nb_type="multinomial", **kwargs)
