"""Naive Bayes Model."""

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from ..base import ClassicalModel


class NaiveBayesModel(ClassicalModel):
    """
    Naive Bayes wrapper supporting Gaussian and Multinomial variants.
    """

    def __init__(self, type="gaussian", **kwargs):
        """
        Initialize the Naive Bayes model.

        Args:
            type (str, optional): 'gaussian' or 'multinomial'. Defaults to "gaussian".
            **kwargs: Additional arguments passed to the underlying sklearn model.
        """
        super().__init__()
        if type == "gaussian":
            self.model = GaussianNB(**kwargs)
        else:
            self.model = MultinomialNB(**kwargs)


class GaussianNaiveBayesModel(NaiveBayesModel):
    """Gaussian Naive Bayes classifier."""

    def __init__(self, **kwargs):
        """Initialize the Gaussian Naive Bayes model."""
        super().__init__(type="gaussian", **kwargs)


class MultinomialNaiveBayesModel(NaiveBayesModel):
    """Multinomial Naive Bayes classifier."""

    def __init__(self, **kwargs):
        """Initialize the Multinomial Naive Bayes model."""
        super().__init__(type="multinomial", **kwargs)
