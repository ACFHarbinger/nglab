"""Linear models facade."""

from .linear.linear_regression import LinearRegressionModel
from .linear.ridge import RidgeRegressionModel
from .linear.lasso import LassoRegressionModel
from .linear.elastic_net import ElasticNetModel
from .linear.lars import LARSModel
from .linear.logistic import LogisticRegressionModel
from .linear.polynomial import PolynomialRegressionModel
from .linear.olsr import OLSRModel
from .linear.stepwise import StepwiseRegressionModel
from .linear.mars import MARSModel
from .linear.loess import LOESSModel

__all__ = [
    "LinearRegressionModel",
    "RidgeRegressionModel",
    "LassoRegressionModel",
    "ElasticNetModel",
    "LARSModel",
    "LogisticRegressionModel",
    "PolynomialRegressionModel",
    "OLSRModel",
    "StepwiseRegressionModel",
    "MARSModel",
    "LOESSModel",
]
