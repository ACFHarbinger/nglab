import torch

from python.src.utils.functions.gpu_features import GPUFeatureEngineer


def test_compute_imbalance():
    engineer = GPUFeatureEngineer(device="cpu")
    bid_v = torch.tensor([100.0, 200.0, 50.0])
    ask_v = torch.tensor([50.0, 200.0, 150.0])
    
    imbalance = engineer.compute_imbalance(bid_v, ask_v)
    
    # (100-50)/(100+50) = 50/150 = 0.333
    # (200-200)/(200+200) = 0
    # (50-150)/(50+150) = -100/200 = -0.5
    
    assert torch.allclose(imbalance, torch.tensor([1/3, 0.0, -0.5]), atol=1e-5)

def test_compute_spread():
    engineer = GPUFeatureEngineer(device="cpu")
    bid_p = torch.tensor([100.0, 105.0])
    ask_p = torch.tensor([101.0, 106.0])
    
    spread = engineer.compute_spread(bid_p, ask_p)
    
    # mid = 100.5, rel_spread = 1/100.5 = 0.00995
    # mid = 105.5, rel_spread = 1/105.5 = 0.00947
    
    assert torch.allclose(spread, torch.tensor([1/100.5, 1/105.5]), atol=1e-5)

def test_compute_vwap():
    engineer = GPUFeatureEngineer(device="cpu")
    prices = torch.tensor([100.0, 101.0, 102.0])
    volumes = torch.tensor([10.0, 20.0, 10.0])
    
    # window=2
    vwap = engineer.compute_vwap(prices, volumes, window=2)
    
    # step 0: mean of [100*10]/[10] = 100 (due to padding/mean logic in moving_average)
    # wait, moving_average returns SMA. 
    # vwap = sum(p*v)/sum(v)
    # moving_average(pv, 2) = [pv[i] + pv[i-1]]/2
    # sum_pv = moving_average(pv, 2) * 2 = pv[i] + pv[i-1]
    
    # idx 1: (100*10 + 101*20) / (10 + 20) = (1000 + 2020) / 30 = 3020 / 30 = 100.666
    assert torch.allclose(vwap[1], torch.tensor(100.6666), atol=1e-3)
