"""Naive Bayes Model."""

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from ..base import ClassicalModel


class NaiveBayesModel(ClassicalModel):
    def __init__(self, type="gaussian", **kwargs):
        super().__init__()
        if type == "gaussian":
            self.model = GaussianNB(**kwargs)
        else:
            self.model = MultinomialNB(**kwargs)


class GaussianNaiveBayesModel(NaiveBayesModel):
    def __init__(self, **kwargs):
        super().__init__(type="gaussian", **kwargs)


class MultinomialNaiveBayesModel(NaiveBayesModel):
    def __init__(self, **kwargs):
        super().__init__(type="multinomial", **kwargs)
