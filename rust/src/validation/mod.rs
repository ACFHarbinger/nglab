/*!
 * Input validation logic for simulation parameters and user orders.
 */

use crate::errors::{ArenaError, ArenaResult};

/// Validates that a price is positive and within reasonable bounds.
pub fn validate_price(price: f64) -> ArenaResult<()> {
    if price <= 0.0 {
        return Err(ArenaError::InvalidPrice(price));
    }
    // Sane upper bound for prices in our simulation (e.g. 1 billion)
    if price > 1_000_000_000.0 {
        return Err(ArenaError::InvalidPrice(price));
    }
    Ok(())
}

/// Validates that a quantity is positive and within limits.
pub fn validate_quantity(quantity: f64, max_limit: Option<f64>) -> ArenaResult<()> {
    if quantity <= 0.0 {
        return Err(ArenaError::InvalidQuantity(quantity));
    }
    if let Some(limit) = max_limit {
        if quantity > limit {
            return Err(ArenaError::InvalidQuantity(quantity));
        }
    }
    Ok(())
}

/// Validates that an asset is known to the system.
pub fn validate_asset(asset: &str, known_assets: &[String]) -> ArenaResult<()> {
    if asset.is_empty() {
        return Err(ArenaError::MarketNotFound(asset.to_string()));
    }
    if !known_assets.contains(&asset.to_string()) {
        return Err(ArenaError::MarketNotFound(asset.to_string()));
    }
    Ok(())
}

/// Validates time-series data length.
pub fn validate_data_length(data: &[f64], min_required: usize) -> ArenaResult<()> {
    if data.len() < min_required {
        return Err(ArenaError::DataLoading(format!(
            "Insufficient data: expected at least {}, found {}",
            min_required,
            data.len()
        )));
    }
    Ok(())
}

/// Validates simulation steps count.
pub fn validate_steps(steps: usize, max_steps: usize) -> ArenaResult<()> {
    if steps == 0 {
        return Err(ArenaError::ValidationError(
            "Steps must be greater than zero".to_string(),
        ));
    }
    if steps > max_steps {
        return Err(ArenaError::ValidationError(format!(
            "Steps {} exceeds limit {}",
            steps, max_steps
        )));
    }
    Ok(())
}

/// Validates forecast horizon.
pub fn validate_horizon(horizon: usize, max_horizon: usize) -> ArenaResult<()> {
    if horizon == 0 {
        return Err(ArenaError::ValidationError(
            "Horizon must be greater than zero".to_string(),
        ));
    }
    if horizon > max_horizon {
        return Err(ArenaError::ValidationError(format!(
            "Horizon {} exceeds limit {}",
            horizon, max_horizon
        )));
    }
    Ok(())
}
