import unittest

import torch

from python.src.policies.black_scholes import BlackScholesPolicy
from python.src.policies.neural import NeuralPolicy
from python.src.policies.regular import RegularPolicy
from python.src.policies.threshold import ThresholdPolicy


class TestPolicies(unittest.TestCase):
    def test_black_scholes(self):
        policy = BlackScholesPolicy()
        # Price 100, Strike 100, TTM 1.0 -> Call should be ~10.45
        # 100 < 10.45 * 0.95 (False)
        # 100 > 10.45 * 1.05 (True) -> Sell (2)
        obs = {"price": 100, "strike": 100, "time_to_maturity": 1.0}
        action = policy.act(obs)
        self.assertEqual(action, 2)

        # Price 1, Strike 100 -> Call ~ 0
        obs = {"price": 1, "strike": 100, "time_to_maturity": 1.0}
        action = policy.act(obs)
        self.assertEqual(action, 2)

    def test_threshold(self):
        policy = ThresholdPolicy({"buy_threshold": 50, "sell_threshold": 150})
        self.assertEqual(policy.act(40), 1)  # Buy
        self.assertEqual(policy.act(160), 2)  # Sell
        self.assertEqual(policy.act(100), 0)  # Hold

    def test_regular(self):
        policy = RegularPolicy({"period": 3})
        # 1: 0
        # 2: 0
        # 3: 1
        self.assertEqual(policy.act(None), 0)
        self.assertEqual(policy.act(None), 0)
        self.assertEqual(policy.act(None), 1)

    def test_neural(self):
        class DummyModel(torch.nn.Module):
            def forward(self, obs):
                return {"action": torch.tensor(1)}

        model = DummyModel()
        policy = NeuralPolicy(model)
        action = policy.act({"foo": "bar"})
        self.assertEqual(action, torch.tensor(1))


if __name__ == "__main__":
    unittest.main()
