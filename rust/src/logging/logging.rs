use opentelemetry::trace::TracerProvider;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{runtime, trace as sdktrace, Resource};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/**
 * Initialize structured logging and distributed tracing.
 *
 * Sets up `tracing` with an `EnvFilter`, file/console output, and OpenTelemetry OTLP exporter.
 */
pub fn init_logging(log_level: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file_appender = tracing_appender::rolling::daily("./logs", "nglab.log");

    // Setup OpenTelemetry OTLP Exporter
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint("http://localhost:4317")
        .build()?;

    let provider = sdktrace::TracerProvider::builder()
        .with_batch_exporter(exporter, runtime::Tokio)
        .with_resource(Resource::new(vec![KeyValue::new(
            "service.name",
            "nglab-engine",
        )]))
        .build();

    let tracer = provider.tracer("nglab-engine");
    global::set_tracer_provider(provider);

    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level)))
        .with(tracing_subscriber::fmt::layer())
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_writer(file_appender),
        )
        .with(telemetry)
        .init();

    Ok(())
}
