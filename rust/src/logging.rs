use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/**
 * Initialize structured logging with file and console output.
 *
 * Sets up `tracing` with an `EnvFilter` and a daily rolling file appender.
 */
pub fn init_logging(log_level: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file_appender = tracing_appender::rolling::daily("./logs", "nglab.log");

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level)))
        .with(tracing_subscriber::fmt::layer())
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_writer(file_appender),
        )
        .init();

    Ok(())
}
