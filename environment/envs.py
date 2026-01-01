"""
Gymnasium-compatible environments wrapping the Rust arena
"""

import numpy as np
import gymnasium as gym

from gymnasium import spaces

# Import Rust bindings (will be available after maturin build)
# For now, we provide a pure-Python fallback
try:
    from nglab._nglab import TradingEnv as RustTradingEnv
    from nglab._nglab import OrderBook as RustOrderBook
    from nglab._nglab import PolymarketArena as RustPolymarketArena
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


class TradingEnv(gym.Env):
    """
    Trading environment for RL agents.
    
    Observation Space:
        Box of shape (lookback, features) containing:
        - Normalized price
        - Returns
        - Volume
        - Order book imbalance
        - Position (normalized)
        - Cash (normalized)
    
    Action Space:
        Discrete(3): 0=Hold, 1=Buy, 2=Sell
    
    Reward:
        Risk-adjusted return with drawdown penalty
    """
    
    metadata: dict[str, object] = {"render_modes": ["human"]}
    
    def __init__(
        self,
        prices: np.ndarray | None = None,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        lookback: int = 30,
        max_steps: int = 1000,
        render_mode: str | None = None,
    ):
        super().__init__()
        
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.lookback = lookback
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Price data
        self.prices = prices if prices is not None else np.random.randn(1000).cumsum() + 100
        
        # Spaces
        self.num_features = 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback, self.num_features),
            dtype=np.float64,
        )
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        
        # State
        self.current_step = lookback
        self.position = 0.0
        self.cash = initial_capital
        self.returns_history: list = []
        self.prev_portfolio_value = initial_capital
        
        # Initialize Rust backend if available
        if HAS_RUST:
            self._rust_env = RustTradingEnv(
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                lookback=lookback,
                max_steps=max_steps,
            )
            self._rust_env.load_prices(self.prices.tolist())
        else:
            self._rust_env = None
    
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        
        if self._rust_env is not None:
            obs = self._rust_env.reset()
            return np.array(obs), {}
        
        # Python fallback
        self.current_step = self.lookback
        self.position = 0.0
        self.cash = self.initial_capital
        self.returns_history = []
        self.prev_portfolio_value = self.initial_capital
        
        return self._get_observation(), {}
    
    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self._rust_env is not None:
            obs, reward, terminated, truncated, info = self._rust_env.step(action)
            return np.array(obs), reward, terminated, truncated, dict(info)
        
        # Python fallback implementation
        trade_cost = self._execute_action(action)
        
        self.current_step += 1
        
        # Calculate return
        portfolio_value = self._portfolio_value()
        ret = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        self.returns_history.append(ret)
        self.prev_portfolio_value = portfolio_value
        
        # Risk-adjusted reward
        reward = ret * 100.0 - trade_cost / self.initial_capital * 100.0
        
        # Termination
        terminated = portfolio_value <= 0.0 or self.current_step >= len(self.prices) - 1
        truncated = self.current_step - self.lookback >= self.max_steps
        
        info = {
            "portfolio_value": portfolio_value,
            "position": self.position,
            "cash": self.cash,
            "sharpe_ratio": self._calculate_sharpe(),
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        obs = np.zeros((self.lookback, self.num_features))
        
        for i in range(self.lookback):
            idx = self.current_step - self.lookback + i
            if idx < len(self.prices):
                price = self.prices[idx]
                prev_price = self.prices[max(0, idx - 1)]
                ret = (price - prev_price) / prev_price if prev_price > 0 else 0.0
                
                obs[i, 0] = price / self.prices[0]  # Normalized price
                obs[i, 1] = ret  # Returns
                obs[i, 2] = 0.0  # Volume placeholder
                obs[i, 3] = 0.0  # Imbalance placeholder
                obs[i, 4] = self.position / self.initial_capital  # Normalized position
                obs[i, 5] = self.cash / self.initial_capital  # Normalized cash
        
        return obs
    
    def _execute_action(self, action: int) -> float:
        if self.current_step >= len(self.prices):
            return 0.0
        
        price = self.prices[self.current_step]
        trade_size = self.initial_capital * 0.1
        cost = 0.0
        
        if action == 1:  # Buy
            shares = trade_size / price
            tx_cost = trade_size * self.transaction_cost
            if self.cash >= trade_size + tx_cost:
                self.cash -= trade_size + tx_cost
                self.position += shares
                cost = tx_cost
        elif action == 2:  # Sell
            if self.position > 0:
                shares_to_sell = min(trade_size / price, self.position)
                proceeds = shares_to_sell * price
                tx_cost = proceeds * self.transaction_cost
                self.position -= shares_to_sell
                self.cash += proceeds - tx_cost
                cost = tx_cost
        
        return cost
    
    def _portfolio_value(self) -> float:
        price = self.prices[min(self.current_step, len(self.prices) - 1)]
        return self.cash + self.position * price
    
    def _calculate_sharpe(self, window: int = 30) -> float:
        if len(self.returns_history) < 2:
            return 0.0
        recent = self.returns_history[-window:]
        if len(recent) < 2:
            return 0.0
        mean = np.mean(recent)
        std = np.std(recent, ddof=1)
        if std > 0:
            return mean / std * np.sqrt(252)
        return 0.0
    
    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, "
                  f"Value: ${self._portfolio_value():.2f}, "
                  f"Position: {self.position:.4f}, "
                  f"Cash: ${self.cash:.2f}")


class ClobEnv(TradingEnv):
    """
    Central Limit Order Book trading environment.
    
    Extends TradingEnv with full order book simulation.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        if HAS_RUST:
            self._orderbook = RustOrderBook()
        else:
            self._orderbook = None


class PolymarketEnv(gym.Env):
    """
    Polymarket prediction market environment.
    
    Observation Space:
        Box containing market prices, positions, and account info
    
    Action Space:
        MultiDiscrete for each market: 0=Hold, 1=Buy Yes, 2=Buy No, 3=Sell Yes, 4=Sell No
    """
    
    metadata: dict[str, object] = {"render_modes": ["human"]}
    
    def __init__(
        self,
        market_ids: list | None = None,
        initial_collateral: float = 10000.0,
        taker_fee: float = 0.001,
        render_mode: str | None = None,
    ):
        super().__init__()
        
        self.initial_collateral = initial_collateral
        self.taker_fee = taker_fee
        self.render_mode = render_mode
        self.market_ids = market_ids or []
        
        num_markets = max(1, len(self.market_ids))
        
        # Observation: [collateral, pnl] + [price, yes_pos, no_pos] per market
        obs_dim = 2 + num_markets * 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float64,
        )
        
        # Action: 5 actions per market
        self.action_space = spaces.MultiDiscrete([5] * num_markets)
        
        if HAS_RUST:
            self._arena = RustPolymarketArena(
                initial_collateral=initial_collateral,
                taker_fee=taker_fee,
            )
        else:
            self._arena = None
            self._collateral = initial_collateral
            self._positions: dict[str, tuple[float, float]] = {}
            self._prices: dict[str, float] = {}
    
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        
        if self._arena is not None:
            # Reset Rust arena
            pass  # Arena reset would be called here
        else:
            self._collateral = self.initial_collateral
            self._positions = {}
            self._prices = {m: 0.5 for m in self.market_ids}
        
        return self._get_observation(), {}
    
    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        # Process actions for each market
        prev_value = self._account_value()
        
        for i, market_id in enumerate(self.market_ids):
            self._execute_market_action(market_id, int(action[i]))
        
        # Calculate reward
        current_value = self._account_value()
        reward = (current_value - prev_value) / self.initial_collateral * 100.0
        
        terminated = current_value <= 0.0
        truncated = False
        
        info = {
            "account_value": current_value,
            "collateral": self._collateral if self._arena is None else self._arena.collateral(),
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        if self._arena is not None:
            collateral = self._arena.collateral()
            pnl = self._arena.realized_pnl()
        else:
            collateral = self._collateral
            pnl = 0.0
        
        obs = [collateral / self.initial_collateral, pnl / self.initial_collateral]
        
        for market_id in self.market_ids:
            if self._arena is not None:
                price = self._arena.get_price(market_id) or 0.5
                yes_pos, no_pos = self._arena.get_position(market_id)
            else:
                price = self._prices.get(market_id, 0.5)
                yes_pos, no_pos = self._positions.get(market_id, (0.0, 0.0))
            
            obs.extend([price, yes_pos / 100.0, no_pos / 100.0])
        
        return np.array(obs)
    
    def _execute_market_action(self, market_id: str, action: int):
        """Execute action: 0=Hold, 1=Buy Yes, 2=Buy No, 3=Sell Yes, 4=Sell No"""
        if action == 0:
            return
        
        amount = self.initial_collateral * 0.01  # 1% position sizing
        
        if self._arena is None:
            # Python fallback
            price = self._prices.get(market_id, 0.5)
            yes_pos, no_pos = self._positions.get(market_id, (0.0, 0.0))
            
            if action == 1:  # Buy Yes
                cost = amount * price * (1 + self.taker_fee)
                if self._collateral >= cost:
                    self._collateral -= cost
                    yes_pos += amount
            elif action == 2:  # Buy No
                cost = amount * (1 - price) * (1 + self.taker_fee)
                if self._collateral >= cost:
                    self._collateral -= cost
                    no_pos += amount
            elif action == 3 and yes_pos >= amount:  # Sell Yes
                proceeds = amount * price * (1 - self.taker_fee)
                self._collateral += proceeds
                yes_pos -= amount
            elif action == 4 and no_pos >= amount:  # Sell No
                proceeds = amount * (1 - price) * (1 - self.taker_fee)
                self._collateral += proceeds
                no_pos -= amount
            
            self._positions[market_id] = (yes_pos, no_pos)
    
    def _account_value(self) -> float:
        if self._arena is not None:
            return self._arena.account_value()
        
        value = self._collateral
        for market_id, (yes_pos, no_pos) in self._positions.items():
            price = self._prices.get(market_id, 0.5)
            value += yes_pos * price + no_pos * (1 - price)
        return value
    
    def render(self):
        if self.render_mode == "human":
            print(f"Account Value: ${self._account_value():.2f}")
