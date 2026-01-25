use serde::{Deserialize, Serialize};

/**
 * Configuration for the circuit breaker.
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Percentage change from base price to trigger a halt
    pub price_threshold_pct: f64,
    /// Number of steps to halt trading for
    pub halt_duration_steps: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            price_threshold_pct: 0.1, // 10%
            halt_duration_steps: 100,
        }
    }
}

/**
 * Circuit breaker to pause trading during extreme volatility.
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    base_price: f64,
    halt_until_step: u64,
}

impl CircuitBreaker {
    /**
     * Create a new circuit breaker.
     */
    pub fn new(config: CircuitBreakerConfig, initial_price: f64) -> Self {
        Self {
            config,
            base_price: initial_price,
            halt_until_step: 0,
        }
    }

    /**
     * Check if a price move triggers a halt.
     *
     * Returns true if trading is halted.
     */
    pub fn check(&mut self, current_price: f64, current_step: u64) -> bool {
        if self.halt_until_step > current_step {
            return true; // Still halted
        }

        let price_diff = (current_price - self.base_price).abs();
        let threshold = self.base_price * self.config.price_threshold_pct;

        if price_diff > threshold {
            self.halt_until_step = current_step + self.config.halt_duration_steps;
            self.base_price = current_price;
            return true;
        }

        false
    }

    /**
     * Reset the circuit breaker with a new base price.
     */
    pub fn reset(&mut self, price: f64) {
        self.base_price = price;
        self.halt_until_step = 0;
    }

    /**
     * Check if trading is currently halted.
     */
    pub fn is_halted(&self, current_step: u64) -> bool {
        self.halt_until_step > current_step
    }
}
