import numpy as np

from python.src.features.regime import MarketRegimeDetector


def test_regime_detector_fit_predict():
    detector = MarketRegimeDetector(n_regimes=2)
    
    # Generate two distinct clusters
    cluster1 = np.random.normal(0, 0.1, (50, 2))
    cluster2 = np.random.normal(5, 0.1, (50, 2))
    X = np.vstack([cluster1, cluster2])
    
    detector.fit(X)
    assert detector.is_fitted
    
    labels = detector.predict_regime(X)
    assert len(labels) == 100
    assert len(np.unique(labels)) == 2
    
    # Check one-hot
    one_hot = detector.get_regime_one_hot(X)
    assert one_hot.shape == (100, 2)
    assert np.all(one_hot.sum(axis=1) == 1)

def test_regime_detector_nan_handling():
    detector = MarketRegimeDetector(n_regimes=2)
    X = np.random.normal(0, 1, (10, 2))
    detector.fit(X)
    
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    
    labels = detector.predict_regime(X_nan)
    assert labels[0] == 0 # Default for NaN
