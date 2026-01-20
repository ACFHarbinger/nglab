use once_cell::sync::Lazy;
use prometheus::{Counter, Histogram, HistogramOpts, IntGauge, Registry};
use std::sync::Arc;

/// Global metrics registry for the NGLab simulation.
pub static REGISTRY: Lazy<Registry> = Lazy::new(Registry::new);

/// Core metrics collected during simulation.
#[derive(Clone)]
pub struct Metrics {
    /// Total number of orders submitted to the system.
    pub orders_total: Counter,
    /// Total number of trades executed in the order book.
    pub trades_total: Counter,
    /// Duration of individual simulation steps in seconds.
    pub step_duration: Histogram,
    /// Current total portfolio value (cash + positions).
    pub portfolio_value: IntGauge,
    /// Current depth of the order book (number of levels).
    pub orderbook_depth: IntGauge,
}

impl Metrics {
    /// Initialize and register metrics with the global registry.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let orders_total = Counter::new("nglab_orders_total", "Total orders submitted")?;
        REGISTRY.register(Box::new(orders_total.clone()))?;

        let trades_total = Counter::new("nglab_trades_total", "Total trades executed")?;
        REGISTRY.register(Box::new(trades_total.clone()))?;

        let step_duration = Histogram::with_opts(
            HistogramOpts::new("nglab_step_duration_seconds", "Step execution time")
                .buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
        )?;
        REGISTRY.register(Box::new(step_duration.clone()))?;

        let portfolio_value = IntGauge::new("nglab_portfolio_value", "Current portfolio value")?;
        REGISTRY.register(Box::new(portfolio_value.clone()))?;

        let orderbook_depth = IntGauge::new("nglab_orderbook_depth", "Order book depth")?;
        REGISTRY.register(Box::new(orderbook_depth.clone()))?;

        Ok(Self {
            orders_total,
            trades_total,
            step_duration,
            portfolio_value,
            orderbook_depth,
        })
    }
}

/// Singleton instance of Metrics.
pub static METRICS: Lazy<Arc<Metrics>> =
    Lazy::new(|| Arc::new(Metrics::new().expect("Failed to initialize metrics")));

/// Export metrics as a Prometheus-formatted string.
pub fn export_metrics() -> String {
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
        return format!("# Error encoding metrics: {}", e);
    }
    String::from_utf8(buffer)
        .unwrap_or_else(|e| format!("# Error converting metrics to UTF-8: {}", e))
}
