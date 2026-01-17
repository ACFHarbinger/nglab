"""
Naive Bayes models.
"""
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from .base import ClassicalModel

class NaiveBayesModel(ClassicalModel):
    def __init__(self, type='gaussian', **kwargs):
        super().__init__()
        if type == 'gaussian':
            self.model = GaussianNB(**kwargs)
        else:
            self.model = MultinomialNB(**kwargs)
