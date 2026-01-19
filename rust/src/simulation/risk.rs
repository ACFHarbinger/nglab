/*!
 * Risk Management System for Trading.
 *
 * Provides position sizing limits, drawdown monitoring, VaR calculation,
 * and automatic risk-based position adjustments. Critical for production trading.
 */

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for position sizing and risk limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Maximum position size as fraction of portfolio (0.0 - 1.0).
    pub max_position_fraction: f64,
    /// Maximum loss allowed per day as fraction of portfolio.
    pub daily_loss_limit: f64,
    /// Maximum drawdown before automatic position reduction.
    pub max_drawdown: f64,
    /// VaR confidence level (e.g., 0.95 for 95% VaR).
    pub var_confidence: f64,
    /// VaR limit as fraction of portfolio.
    pub var_limit: f64,
    /// Lookback period for VaR calculation (in observations).
    pub var_lookback: usize,
    /// Enable automatic position reduction on limit breaches.
    pub auto_reduce_positions: bool,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_position_fraction: 0.10, // Max 10% in single position
            daily_loss_limit: 0.02,      // Max 2% daily loss
            max_drawdown: 0.15,          // Max 15% drawdown
            var_confidence: 0.95,        // 95% VaR
            var_limit: 0.05,             // Max 5% VaR
            var_lookback: 252,           // ~1 year of daily data
            auto_reduce_positions: true,
        }
    }
}

/// Current risk status and metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskStatus {
    /// Current Value at Risk.
    pub current_var: f64,
    /// Current drawdown from peak.
    pub current_drawdown: f64,
    /// Daily P&L.
    pub daily_pnl: f64,
    /// Is daily loss limit breached?
    pub daily_limit_breached: bool,
    /// Is drawdown limit breached?
    pub drawdown_breached: bool,
    /// Is VaR limit breached?
    pub var_breached: bool,
    /// Overall risk level (0-100).
    pub risk_score: u8,
    /// Recommended position size multiplier (0.0 - 1.0).
    pub position_multiplier: f64,
}

impl Default for RiskStatus {
    fn default() -> Self {
        Self {
            current_var: 0.0,
            current_drawdown: 0.0,
            daily_pnl: 0.0,
            daily_limit_breached: false,
            drawdown_breached: false,
            var_breached: false,
            risk_score: 0,
            position_multiplier: 1.0,
        }
    }
}

/// Risk manager tracking portfolio risk metrics.
#[derive(Debug, Clone)]
pub struct RiskManager {
    /// Risk configuration.
    config: RiskConfig,
    /// Historical returns for VaR calculation.
    returns_history: VecDeque<f64>,
    /// Peak portfolio value.
    peak_value: f64,
    /// Current portfolio value.
    current_value: f64,
    /// Daily starting value.
    daily_start_value: f64,
    /// Current risk status.
    status: RiskStatus,
}

impl RiskManager {
    /// Create a new risk manager with the given configuration.
    pub fn new(config: RiskConfig, initial_capital: f64) -> Self {
        Self {
            config,
            returns_history: VecDeque::with_capacity(500),
            peak_value: initial_capital,
            current_value: initial_capital,
            daily_start_value: initial_capital,
            status: RiskStatus::default(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(initial_capital: f64) -> Self {
        Self::new(RiskConfig::default(), initial_capital)
    }

    /// Update portfolio value and recalculate risk metrics.
    pub fn update(&mut self, portfolio_value: f64) {
        let prev_value = self.current_value;
        self.current_value = portfolio_value;

        // Calculate return
        if prev_value > 0.0 {
            let ret = (portfolio_value - prev_value) / prev_value;
            self.returns_history.push_back(ret);

            // Maintain lookback window
            if self.returns_history.len() > self.config.var_lookback {
                self.returns_history.pop_front();
            }
        }

        // Update peak value
        if portfolio_value > self.peak_value {
            self.peak_value = portfolio_value;
        }

        // Recalculate all risk metrics
        self.calculate_metrics();
    }

    /// Start a new trading day (reset daily P&L tracking).
    pub fn new_trading_day(&mut self) {
        self.daily_start_value = self.current_value;
    }

    /// Calculate all risk metrics.
    fn calculate_metrics(&mut self) {
        // Drawdown
        self.status.current_drawdown = if self.peak_value > 0.0 {
            (self.peak_value - self.current_value) / self.peak_value
        } else {
            0.0
        };

        // Daily P&L
        self.status.daily_pnl = if self.daily_start_value > 0.0 {
            (self.current_value - self.daily_start_value) / self.daily_start_value
        } else {
            0.0
        };

        // VaR calculation
        self.status.current_var = self.calculate_var();

        // Check limit breaches
        self.status.daily_limit_breached = -self.status.daily_pnl > self.config.daily_loss_limit;
        self.status.drawdown_breached = self.status.current_drawdown > self.config.max_drawdown;
        self.status.var_breached = self.status.current_var > self.config.var_limit;

        // Calculate overall risk score (0-100)
        self.status.risk_score = self.calculate_risk_score();

        // Calculate position multiplier
        self.status.position_multiplier = self.calculate_position_multiplier();
    }

    /// Calculate Value at Risk using historical simulation.
    fn calculate_var(&self) -> f64 {
        if self.returns_history.len() < 10 {
            return 0.0;
        }

        let mut sorted_returns: Vec<f64> = self.returns_history.iter().cloned().collect();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Find the VaR percentile
        let percentile_idx =
            ((1.0 - self.config.var_confidence) * sorted_returns.len() as f64).floor() as usize;
        let var = -sorted_returns.get(percentile_idx).copied().unwrap_or(0.0);

        var.max(0.0)
    }

    /// Calculate overall risk score (0-100).
    fn calculate_risk_score(&self) -> u8 {
        let mut score = 0.0;

        // Drawdown contribution (0-40 points)
        let dd_ratio = self.status.current_drawdown / self.config.max_drawdown;
        score += (dd_ratio * 40.0).min(40.0);

        // Daily loss contribution (0-30 points)
        let daily_ratio = (-self.status.daily_pnl) / self.config.daily_loss_limit;
        score += (daily_ratio * 30.0).min(30.0).max(0.0);

        // VaR contribution (0-30 points)
        let var_ratio = self.status.current_var / self.config.var_limit;
        score += (var_ratio * 30.0).min(30.0);

        score.min(100.0) as u8
    }

    /// Calculate position size multiplier based on risk.
    fn calculate_position_multiplier(&self) -> f64 {
        if self.status.daily_limit_breached {
            return 0.0; // No new positions after daily limit breach
        }

        if self.status.drawdown_breached {
            return 0.25; // Heavily reduced positions
        }

        if self.status.var_breached {
            return 0.5; // Moderately reduced positions
        }

        // Gradual reduction based on risk score
        match self.status.risk_score {
            0..=25 => 1.0,
            26..=50 => 0.9,
            51..=75 => 0.7,
            76..=100 => 0.5,
            _ => 1.0,
        }
    }

    /// Get the current risk status.
    pub fn status(&self) -> &RiskStatus {
        &self.status
    }

    /// Get the current risk configuration.
    pub fn config(&self) -> &RiskConfig {
        &self.config
    }

    /// Check if trading should be halted.
    pub fn should_halt_trading(&self) -> bool {
        self.status.daily_limit_breached
    }

    /// Check if position size should be reduced.
    pub fn should_reduce_positions(&self) -> bool {
        self.config.auto_reduce_positions
            && (self.status.drawdown_breached || self.status.var_breached)
    }

    /// Get the recommended maximum position size.
    pub fn max_position_size(&self) -> f64 {
        self.current_value * self.config.max_position_fraction * self.status.position_multiplier
    }

    /// Calculate Sharpe ratio from historical returns.
    pub fn sharpe_ratio(&self) -> f64 {
        if self.returns_history.len() < 30 {
            return 0.0;
        }

        let returns: Vec<f64> = self.returns_history.iter().cloned().collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            // Annualized Sharpe (assuming daily returns, 252 trading days)
            (mean * 252.0_f64.sqrt()) / std_dev
        } else {
            0.0
        }
    }

    /// Calculate Sortino ratio (downside deviation only).
    pub fn sortino_ratio(&self) -> f64 {
        if self.returns_history.len() < 30 {
            return 0.0;
        }

        let returns: Vec<f64> = self.returns_history.iter().cloned().collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;

        // Downside deviation
        let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        if negative_returns.is_empty() {
            return f64::INFINITY;
        }

        let downside_variance: f64 =
            negative_returns.iter().map(|r| r.powi(2)).sum::<f64>() / negative_returns.len() as f64;
        let downside_std = downside_variance.sqrt();

        if downside_std > 0.0 {
            (mean * 252.0_f64.sqrt()) / downside_std
        } else {
            0.0
        }
    }

    /// Reset the risk manager to initial state.
    pub fn reset(&mut self, initial_capital: f64) {
        self.returns_history.clear();
        self.peak_value = initial_capital;
        self.current_value = initial_capital;
        self.daily_start_value = initial_capital;
        self.status = RiskStatus::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_manager_creation() {
        let rm = RiskManager::with_defaults(100_000.0);
        assert_eq!(rm.current_value, 100_000.0);
        assert_eq!(rm.peak_value, 100_000.0);
        assert!(!rm.should_halt_trading());
    }

    #[test]
    fn test_drawdown_calculation() {
        let mut rm = RiskManager::with_defaults(100_000.0);

        // Simulate a 10% loss
        rm.update(90_000.0);

        assert!((rm.status().current_drawdown - 0.10).abs() < 0.001);
    }

    #[test]
    fn test_daily_loss_limit() {
        let mut rm = RiskManager::with_defaults(100_000.0);

        // Simulate a 3% daily loss (exceeds 2% limit)
        rm.update(97_000.0);

        assert!(rm.status().daily_limit_breached);
        assert!(rm.should_halt_trading());
        assert_eq!(rm.status().position_multiplier, 0.0);
    }

    #[test]
    fn test_max_drawdown_breach() {
        let mut rm = RiskManager::with_defaults(100_000.0);
        rm.new_trading_day(); // Reset daily tracking

        // Simulate 20% drawdown (exceeds 15% limit)
        rm.update(80_000.0);
        rm.new_trading_day(); // New day to avoid daily limit
        rm.update(80_000.0);

        assert!(rm.status().drawdown_breached);
        assert!(rm.should_reduce_positions());
    }

    #[test]
    fn test_var_calculation() {
        let mut rm = RiskManager::with_defaults(100_000.0);

        // Add some return history
        for i in 0..100 {
            let change = if i % 3 == 0 { -0.01 } else { 0.005 };
            let new_value = rm.current_value * (1.0 + change);
            rm.update(new_value);
        }

        // VaR should be calculated
        assert!(rm.status().current_var > 0.0);
    }

    #[test]
    fn test_position_multiplier_scaling() {
        let rm = RiskManager::with_defaults(100_000.0);

        // Default should be full position
        assert_eq!(rm.status().position_multiplier, 1.0);
        assert_eq!(rm.max_position_size(), 10_000.0); // 10% of 100k
    }

    #[test]
    fn test_sharpe_ratio() {
        let mut rm = RiskManager::with_defaults(100_000.0);

        // Not enough data
        assert_eq!(rm.sharpe_ratio(), 0.0);

        // Add consistent positive returns
        for _ in 0..50 {
            let new_value = rm.current_value * 1.001; // 0.1% daily gain
            rm.update(new_value);
        }

        // Should have positive Sharpe
        assert!(rm.sharpe_ratio() > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut rm = RiskManager::with_defaults(100_000.0);

        // Make some updates
        rm.update(90_000.0);
        rm.update(85_000.0);

        // Reset
        rm.reset(100_000.0);

        assert_eq!(rm.current_value, 100_000.0);
        assert_eq!(rm.peak_value, 100_000.0);
        assert_eq!(rm.status().current_drawdown, 0.0);
    }
}
