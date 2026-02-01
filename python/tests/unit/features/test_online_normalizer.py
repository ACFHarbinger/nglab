import numpy as np

from python.src.features.normalization import OnlineNormalizer


def test_online_normalizer_welford():
    normalizer = OnlineNormalizer(feature_dim=1)
    data = np.array([10.0, 20.0, 30.0])
    
    for x in data:
        normalizer.update(np.array([x]))
        
    assert normalizer.n == 3
    assert np.isclose(normalizer.mean[0], 20.0)
    # Variance = [(10-20)^2 + (20-20)^2 + (30-20)^2] / 3 = [100 + 0 + 100] / 3 = 66.666
    assert np.isclose(normalizer.m2[0], 200.0)
    
    # Transform 20 should be 0
    scaled = normalizer.transform(np.array([20.0]))
    assert np.isclose(scaled[0], 0.0)
    
    # Transform 30
    std = np.sqrt(200.0 / 3) # ~8.165
    scaled = normalizer.transform(np.array([30.0]))
    assert np.isclose(scaled[0], (30-20)/std)

def test_online_normalizer_multidim():
    normalizer = OnlineNormalizer(feature_dim=2)
    data = np.array([[10, 100], [20, 200]])
    
    for x in data:
        normalizer.update(x)
        
    assert np.allclose(normalizer.mean, [15, 150])
