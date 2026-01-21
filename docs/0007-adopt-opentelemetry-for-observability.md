# ADR-0007: Adopt OpenTelemetry for Observability

## Status
Accepted

## Context
NGLab is a complex distributed system involving a Rust simulation engine, Python ML processes, and a Tauri frontend. When a trade fails or a simulation behaves unexpectedly, it is difficult to trace the root cause across these process boundaries using disjoint log files. We need a unified way to visualize the flow of execution and performance bottlenecks.

## Decision
We will adopt **OpenTelemetry (OTLP)** as our observability standard.
- **Rust**: We will use the `tracing` crate ecosystem (`tracing-opentelemetry`, `opentelemetry-otlp`) to collect logs and spans.
- **Python**: We will use the `opentelemetry-sdk` to instrument Python code.
- **Export**: All components will export traces to a local OTLP collector (e.g., Jaeger or a generic OTLP endpoint) running via Docker.

## Consequences
- **Easier**:
    - **Distributed Debugging**: We can see a "Waterfall" view of a single order as it flows from UI -> Rust -> Python -> Rust -> UI.
    - **Performance Profiling**: Spans automatically capture execution duration, helping identify slow code paths.
    - **Standardization**: OTLP is vendor-neutral; we can switch backends (Jaeger, Honeycomb, Prometheus) without code changes.
- **Difficult**:
    - **Infrastructure**: Requires running a sidecar collector (Jaeger) for local development.
    - **Volume**: Tracing generates significantly more data than logging. We must configure appropriate sampling rates and `EnvFilter` levels to avoid performance regression.

## Alternatives Considered
- **Plain Logging (log/env_logger)**: Insufficient for understanding causal relationships between async tasks and cross-process calls.
- **Prometheus Metrics Only**: Good for aggregates (counters, gauges) but lacks the context (trace ID) needed to debug individual request failures.
- **Proprietary Agents (Datadog/NewRelic)**: Too heavy and cost-prohibitive for a local-first application.
