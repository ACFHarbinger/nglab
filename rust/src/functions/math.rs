/*!
 * Robust mathematical utilities for financial simulation.
 */

/// A wrapper for f64 that prevents NaN and Infinity.
///
/// Useful for neural network inputs and observation normalization.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SafeFloat(f64);

impl SafeFloat {
    /// Creates a new SafeFloat, handling NaN and Infinity.
    pub fn new(val: f64) -> Self {
        if val.is_nan() {
            Self(0.0)
        } else if val.is_infinite() {
            if val.is_sign_positive() {
                Self(f64::MAX)
            } else {
                Self(f64::MIN)
            }
        } else {
            Self(val)
        }
    }

    /// Returns the underlying f64 value.
    pub fn unwrap(self) -> f64 {
        self.0
    }
}

/// Robust division with zero-check and fallback.
pub fn safe_div(numerator: f64, denominator: f64, fallback: f64) -> f64 {
    if denominator.abs() < 1e-12 {
        fallback
    } else {
        numerator / denominator
    }
}

/// Sane normalization that prevents division by zero in zero-variance data.
pub fn robust_normalize(val: f64, mean: f64, std: f64) -> f64 {
    if std < 1e-12 {
        0.0
    } else {
        (val - mean) / std
    }
}
