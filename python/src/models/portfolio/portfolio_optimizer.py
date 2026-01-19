"""
Portfolio Optimization algorithms for multi-asset management.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

class PortfolioOptimizer:
    """
    Graduates from simple environment-level actions to holistic portfolio management.
    """

    @staticmethod
    def mean_variance_optimization(
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        target_return: Optional[float] = None,
        constraints: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Markowitz Mean-Variance Optimization.
        Minimize: w^T * Sigma * w - lambda * w^T * mu
        Subject to: sum(w) = 1, w >= 0
        """
        n = len(expected_returns)
        
        def objective(w):
            port_variance = np.dot(w.T, np.dot(covariance_matrix, w))
            port_return = np.dot(w, expected_returns)
            return port_variance - risk_aversion * port_return

        initial_weights = np.array([1.0 / n] * n)
        bounds = [(0, 1) for _ in range(n)]
        
        # Default constraints: sum(weights) == 1
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        if constraints:
            cons.extend(constraints)
            
        if target_return is not None:
            cons.append({"type": "eq", "fun": lambda w: np.dot(w, expected_returns) - target_return})

        result = minimize(objective, initial_weights, method="SLSQP", bounds=bounds, constraints=cons)
        
        if not result.success:
            print(f"MVO optimization failed: {result.message}. Returning equal weights.")
            return initial_weights
            
        return result.x

    @staticmethod
    def hierarchical_risk_parity(covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Hierarchical Risk Parity (HRP) algorithm.
        Robust to unstable covariance matrices by using hierarchical clustering.
        """
        # 1. Quasi-Diagonalization
        corr = PortfolioOptimizer._cov_to_corr(covariance_matrix)
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0) # Force zero diagonal for numerical stability
        link = linkage(squareform(dist), method="single")
        indices = leaves_list(link)
        
        # 2. Recursive Bisection
        weights = pd.Series(1.0, index=indices)
        items = [indices.tolist()]
        
        while len(items) > 0:
            items = [i for j in items for i in PortfolioOptimizer._bisect(j, covariance_matrix, weights)]
            
        return weights.sort_index().values

    @staticmethod
    def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.diag(cov))
        return cov / np.outer(std, std)

    @staticmethod
    def _bisect(indices: List[int], cov: np.ndarray, weights: pd.Series) -> List[List[int]]:
        if len(indices) <= 1:
            return []
            
        # Split into two clusters
        mid = len(indices) // 2
        left = indices[:mid]
        right = indices[mid:]
        
        # Calculate cluster variances
        var_l = PortfolioOptimizer._get_cluster_var(cov, left)
        var_r = PortfolioOptimizer._get_cluster_var(cov, right)
        
        # Allocation factor
        alpha = 1 - var_l / (var_l + var_r)
        
        # Update weights
        weights.loc[left] *= alpha
        weights.loc[right] *= (1 - alpha)
        
        return [left, right]

    @staticmethod
    def _get_cluster_var(cov: np.ndarray, indices: List[int]) -> float:
        cluster_cov = cov[np.ix_(indices, indices)]
        # Inverse variance weights within cluster
        w = 1.0 / np.diag(cluster_cov)
        w /= w.sum()
        return np.dot(w.T, np.dot(cluster_cov, w))

    @staticmethod
    def risk_parity_allocation(covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Risk Parity (Inverse Volatility) allocation.
        Weights assets such that each contributes equally to portfolio risk (volatility).
        For diag covariance, this is Inverse Variance weighting.
        """
        # Calculate asset-level volatilities
        vols = np.sqrt(np.diag(covariance_matrix))
        # Inverse volatility weighting
        inv_vols = 1.0 / np.maximum(vols, 1e-8)
        weights = inv_vols / np.sum(inv_vols)
        return weights
