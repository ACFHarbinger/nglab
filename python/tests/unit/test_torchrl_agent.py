import unittest
from typing import cast

from tensordict import TensorDict

from python.src.env.env_wrapper import TradingEnvWrapper
from python.src.env.trading_env import TradingEnv


class TestTorchRLAgent(unittest.TestCase):
    def setUp(self):
        self.env = TradingEnvWrapper(TradingEnv(feature_dim=4))

    def test_env_specs(self):
        # Check specs
        self.assertIsNotNone(self.env.action_spec)
        self.assertIsNotNone(self.env.observation_spec)

    def test_step(self):
        tensordict = self.env.reset()
        self.assertIsInstance(tensordict, TensorDict)
        self.assertTrue("observation" in tensordict.keys())
        self.assertEqual(tensordict["observation"].shape[-1], 4)

        # Step with random action
        action = self.env.action_spec.sample()
        tensordict["action"] = action
        next_td = self.env.step(tensordict)

        # TorchRL GymWrapper typically puts resulting state in 'next'
        self.assertTrue("next" in next_td.keys())

        # Explicitly cast to TensorDict for type checking
        next_state = cast(TensorDict, next_td["next"])

        self.assertTrue("reward" in next_state.keys())
        self.assertTrue(
            "done" in next_state.keys() or "terminated" in next_state.keys()
        )


if __name__ == "__main__":
    unittest.main()
