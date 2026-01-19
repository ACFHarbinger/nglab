# ADR-0002: PyTorch as Core Deep Learning Framework

## Status
Accepted

## Context
The project requires a flexible and performant deep learning framework that supports both reinforcement learning and complex time-series architectures (Transformers, LSTM, etc.).

## Decision
We will use PyTorch as our primary deep learning framework. It has a dominant ecosystem in research, excellent GPU acceleration support, and a more intuitive imperative programming model compared to alternatives.

## Consequences
- **Easier**: Rapid prototyping, access to a vast library of pre-trained models and state-of-the-art research implementations.
- **Difficult**: Deployment can be complex due to large binary sizes and dependency management.

## Alternatives Considered
- **TensorFlow/Keras**: Stronger deployment tooling (TF Serving), but less flexible for custom RL research and more complex to debug.
- **JAX**: Excellent for high-performance functional transformations, but has a smaller ecosystem for high-level RL and time-series abstractions.
