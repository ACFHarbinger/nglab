# Changelog

All notable changes to the NGLab project will be documented in this file.

## [Unreleased] - 2026-01-17

### Added

- **Comprehensive Documentation**:
  - Added high-fidelity JSDoc to the entire TypeScript frontend, including React components (Tabs, Dashboard, Terminal), custom hooks, and utility functions.
  - Converted Rust codebase documentation to standard inner-module style (`//!`) and added detailed item-level documentation for all structs, enums, traits, and functions in `rust/src/` and `rust/benchmarks/`.
  - Added module-level documentation to CSS files (`App.css`, `index.css`) to improve stylistic transparency.
  - Documented the Python Gymnasium environment wrappers in `environment/` using PEP 257 docstrings; verified compliance with `check_docstrings.py`.
  - Added top-level package initialization documentation to `nglab/__init__.py`.
  - Documented all utility scripts in `scripts/` with detailed headers for usage and maintenance.
- **Classical Machine Learning Models (Expanded)**:
  - **Regression**: Added LARS, Stepwise, M5, MARS, LOESS, and classical linear variants.
  - **Decision Trees**: Added comprehensive suite including CART, ID3, C4.5, C5.0, CHAID, DecisionStump, and ConditionalTree.
  - **Ensemble Methods**: Implemented AdaBoost, Bagging, Stacking, Voting, WeightedAverage (Blending), and GBRT.
  - **Bayesian**: Added GaussianNB, MultinomialNB, AODE, and BayesianNetwork (BBN).
  - **SVM Variants**: Added LinearSVM, NuSVM, OneClassSVM (Anomaly), LS-SVM, and Twin SVM (TWSVM).
  - **Clustering**: Added K-Medians (custom L1) and Expectation Maximisation (EM via GMM).
  - **Association Rules**: Added Eclat algorithm for vertical itemset mining.
  - **Association Rule Learning**: Added `EclatAlgorithm` (custom) alongside `Apriori` and `FPGrowth`.
  - **Dimensionality Reduction (Expanded)**: Added `PCR`, `PLSR`, `MDS`, `Sammon Mapping` (custom), `Projection Pursuit` (FastICA), `QDA`, `MDA` (custom), `FDA` (MARS-based), and `UMAP` (wrapper).
  - **Integration**: All models wrapped in `ClassicalModel` and integrated into `TimeSeriesBackbone` or `HelperModelFactory` for seamless PyTorch interoperability.
- **CI/CD & Code Quality**:
  - Integrated a comprehensive quality suite with `Black`, `Ruff`, `MyPy`, `Pip-Audit`, and `Pytest-Cov` for Python.
  - Re-integrated `cargo fmt` and `cargo clippy` for Rust quality assurance.
  - Added `Prettier` for TypeScript and JavaScript formatting with strict isolation.
  - Integrated `ktlint` via Gradle for the Android Kotlin codebase.
  - Implemented strict **Language Isolation** and directory exclusions in `pre-commit` to prevent unintentional changes to Markdown, Kotlin, and metadata files.
  - Standardized `Ruff` configuration to use the modern `[tool.ruff.lint]` structure and updated `MyPy` to Python 3.10 support.
  - Implemented `HelperModelFactory` for unified access to supplemental ML algorithms.
- **Project Structure & Dependencies**:
  - Added `scikit-learn`, `xgboost`, and `lightgbm` to `pyproject.toml`.
  - Created modularized test fixtures in `python/tests/fixtures/` (`deep_fixtures.py`, `mac_fixtures.py`).
  - Added `ktlint` Gradle plugin support to the Android module.

### Changed

- **Tauri Backend Refactoring**:
  - Modularized the Tauri `lib.rs` into specialized submodules for state management (`state.rs`) and categorized command handlers (`commands/`).
- **Project Structure**: Improved maintainability of `typescript/src-tauri/src/` by decoupling commands from the main library entry point.
- **Test Infrastructure**: Updated `conftest.py` with global fixture loading and root path resolution for Python tests.

### Fixed

- **Build Environment**: Resolved linker issues in `.cargo/config.toml` to support `cargo doc` and standard builds in heterogeneous environments.
- **Deep Learning Models**: Fixed relative imports and crashing bugs in `nstansformer`, `tsmamba`, and `xlstm`.
- **Test Stability**: Resolved shape mismatches in `TimeSeriesVAE` tests and `NameError` in classical model fitting unit tests.
- **Code Standards**: Fixed wildcard imports in Android Kotlin tests and adapters to comply with `ktlint` standards.
- **Python Quality**: Addressed various import errors and shape mismatches identified during the CI integration of classical and deep models.

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
  - **Stacked Auto-Encoders (SAE)**: Deep AutoEncoder composed of stacked shallow AEs.
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
