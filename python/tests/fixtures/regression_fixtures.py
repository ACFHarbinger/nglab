import pytest
import torch


@pytest.fixture
def regression_data():
    X = torch.randn(50, 5)
    y = 2 * X[:, 0] - 3 * X[:, 1] + 0.5 * torch.randn(50)
    return X, y
