import unittest

import numpy as np

from python.src.env.trading_env import TradingEnv


class TestTradingEnv(unittest.TestCase):
    def test_init(self):
        env = TradingEnv(lookback=20, max_steps=500, feature_dim=10)
        self.assertEqual(env.lookback, 20)
        self.assertEqual(env.max_steps, 500)
        self.assertEqual(env.feature_dim, 10)
        self.assertEqual(env.current_step, 0)

    def test_action_space(self):
        env = TradingEnv()
        action = env.action_space.sample()
        self.assertEqual(action.shape, (1,))
        self.assertTrue(-1 <= action[0] <= 1)

    def test_observation_space(self):
        env = TradingEnv(feature_dim=15)
        obs = env.observation_space.sample()
        self.assertEqual(obs.shape, (15,))

    def test_reset(self):
        env = TradingEnv()
        obs, info = env.reset()
        self.assertEqual(obs.shape, (12,))
        self.assertIsInstance(info, dict)
        self.assertEqual(env.current_step, 0)

    def test_step(self):
        env = TradingEnv(max_steps=10)
        env.reset()
        action = np.array([0.5])
        obs, reward, terminated, truncated, info = env.step(action)

        self.assertEqual(obs.shape, (12,))
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIsInstance(info, dict)
        self.assertEqual(env.current_step, 1)

    def test_step_termination(self):
        env = TradingEnv(max_steps=3)
        env.reset()

        for i in range(3):
            action = np.array([0.0])
            _, _, terminated, _, _ = env.step(action)
            if i < 2:
                self.assertFalse(terminated)
            else:
                self.assertTrue(terminated)

    def test_render(self):
        env = TradingEnv()
        # Should not raise error
        env.render()

    def test_close(self):
        env = TradingEnv()
        # Should not raise error
        env.close()
