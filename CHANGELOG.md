# Changelog

All notable changes to the NGLab project will be documented in this file.

## [Unreleased] - 2026-01-16

### Added
- **Neural Network Architectures (Advanced)**:
    - **Perceptron (P)**: Basic single-layer feedforward network with configurable activations.
    - **Markov Chain (MC)**: Probabilistic state transition model with learnable matrices.
    - **Boltzmann Machine (BM)**: Stochastic recurrent network with symmetric connections and energy-based learning.
    - **Deep Belief Network (DBN)**: Stack of RBMs with greedy layer-wise training.
    - **Deep Convolutional Network (DCN)**: Hierarchical CNN with BN and pooling for deep features.
    - **Deconvolutional Network (DN)**: Transposed convolutions for upsampling and reconstruction.
    - **Deep Convolutional Inverse Graphics Network (DCIGN)**: Disentangled representation learning (pose, lighting vs identity).
    - **Liquid State Machine (LSM)**: Reservoir computing with spiking neurons and fixed sparse recurrence.
    - **Deep Residual Network (DRN)**: Residual blocks with skip connections for very deep training.
    - **Differentiable Neural Computer (DNC)**: External addressable memory with content/temporal linkage.
    - **Neural Turing Machine (NTM)**: Addressable external memory with shift/sharpening mechanisms.
    - **Attention Network (AN)**: Multi-head self-attention mechanism with positional encoding.
    - **Normalizing Flow (Flow)**: RealNVP-based generative model with invertible affine coupling layers.
    - **Neural ODE (NODE)**: Continuous-time depth model with RK4 solver.
    - **Physics-Informed Neural Network (PINN)**: MLP with gradient supervision for PDE solving.
- **Neural Network Architectures (Standard)**:
    - Implemented Spiking Neural Network (SNN) with custom `LIFCell`.
    - Added MLP, RBF, AE, DAE, SAE, Hopfield Network, ESN, ELM, KohonenMap (SOM), and Capsule Layers.
    - Implemented Rolling Window CNN, TimeGAN, and Diffusion U-Net (1D).
- **Integration**:
    - Fully integrated all new models into the `TimeSeriesBackbone` factory.
    - Added support for `output_type` ('prediction' vs 'embedding') across all backbone models.
    - Added `return_sequence` support for all applicable architectures.
- **Frontend / Dashboard**:
    - **News Tab**: New tab for aggregating news feeds from customized sources (Crypto, Social, Market Data).
    - **Training Tab**: Dedicated interface for configuring and training neural network models directly from the UI.
    - **Prediction Tab**: Added "Deep Learning" model selection to run inference with pre-trained PyTorch models.
    - **Dashboard UI**: Refined `UserProfileWidget` with "Profile Stats" design and improved PnL charts (dynamic coloring, sparkline style).
    - **Navigation**: optimizing tab ordering for better workflow (News moved to end).
- **Backend (Rust & Python)**:
    - **Commands**: Added `list_trained_models` and `predict_trained_model` Tauri commands.
    - **Inference**: Created `infer.py` for standalone model inference via subprocess.
    - **Refactoring**: Reorganized deep learning models into `python/src/models/deep/` for better structure.
- **CI/CD**:
    - Created GitHub Actions workflow for automated Python, Rust, and TypeScript testing/linting.

### Fixed
- Standardized RNN (LSTM/GRU) and xLSTM interfaces for backbone compatibility.
- Fixed VAE KL-annealing logic and reconstruction shape mismatches.
- Resolved various import and type-check issues in the Python pipeline.

### Changed
- **Modularity**: Split the model library into individual specialized files for better maintainability.
- **Testing**: Consolidated architecture tests into a structured, class-based suite in `test_architectures.py`.
- Updated `walkthrough.md` with comprehensive documentation for all new architectures.
