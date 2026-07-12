# ADR-0011: Use ts-rs for Type Generation

## Status
Accepted

## Context
Maintaining synchronization between Rust structs (backend) and TypeScript interfaces (frontend) manually is error-prone. A mismatch (e.g., a field rename or type change) can lead to runtime errors in serialization/deserialization that are not caught by the compiler.

## Decision
We will use the **`ts-rs`** crate to automate the generation of TypeScript type definitions.
- **Annotation**: Rust structs exposed to the frontend will be decorated with `#[derive(TS)]`.
- **Generation**: Running `cargo test` (or a dedicated script) will output corresponding `.ts` files to `typescript/src/bindings/`.
- **Consumption**: The frontend imports these generated types directly.

## Consequences
- **Easier**:
    - **Safety**: "Single source of truth" in Rust. Changes in backend types automatically break the frontend build if incompatible.
    - **Productivity**: Developers don't need to write TypeScript interfaces manually.
- **Difficult**:
    - **Setup**: Cannot handle some complex Rust types or custom serializers automatically without manual overrides.
    - **Workflow**: Requires running the generation step (e.g., via cargo test) before working on the frontend if backend types changed.

## Alternatives Considered
- **Manual Interfaces**: Too fragile and labor-intensive.
- **OpenAPI/Swagger**: Generating types from an OpenAPI spec. Good for HTTP APIs, but we use Tauri IPC/WebSockets, and `ts-rs` is lighter weight for direct struct mapping.
- **Protobuf/gRPC**: Enforces strict schema but introduces a heavy serialization layer and compilation step that feels "non-native" for a web frontend.
