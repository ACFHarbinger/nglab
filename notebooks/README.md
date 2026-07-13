# NGLab Tutorial Notebooks

This directory contains interactive Jupyter notebooks that guide you through NGLab's architecture and capabilities.

## 📚 Learning Path

The notebooks are designed to be completed in order, building upon concepts from previous lessons:

### Beginner Track (Notebooks 1-3)
1. **01_quickstart_introduction.ipynb** - NGLab philosophy, architecture, and component communication
2. **02_rust_orderbook_basics.ipynb** - Order matching engine, risk management, VaR calculation
3. **03_trading_environment.ipynb** - Gymnasium interface, step lifecycle, random agent baseline

### Intermediate Track (Notebooks 4-7)
4. **04_time_series_forecasting.ipynb** - ARIMA, GARCH, Prophet models, TSMamba architecture
5. **05_deep_learning_models.ipynb** - VAE market regime detection, TCN vs Transformers
6. **06_hyperparameter_optimization.ipynb** - DEHB algorithm, search spaces, optimization
7. **07_reinforcement_learning_training.ipynb** - PPO training, TensorBoard monitoring

### Advanced Track (Notebooks 8-10)
8. **08_multi_agent_simulation.ipynb** - Competitive environments, Nash equilibria
9. **09_backtesting_framework.ipynb** - Historical evaluation, performance metrics
10. **10_advanced_topics.ipynb** - Custom rewards, model ensembles, uncertainty quantification

## 🚀 Getting Started

### Prerequisites

```bash
# Install NGLab package
cd ../rust && maturin develop

# Install Python dependencies
pip install jupyter matplotlib seaborn pandas

# Launch Jupyter
jupyter notebook
```

### Quick Start

```python
# In any notebook cell
import nglab
import numpy as np
import matplotlib.pyplot as plt

# Verify installation
env = nglab.TradingEnv()
obs, info = env.reset()
print(f"Environment ready! Observation shape: {obs.shape}")
```

## 📖 Notebook Structure

Each notebook follows this structure:
- **Learning Objectives**: What you'll master
- **Theory**: Mathematical foundations and algorithms
- **Code Examples**: Interactive, runnable code
- **Visualizations**: Charts and plots for intuition
- **Exercises**: Optional challenges (marked with 🎯)
- **Summary**: Key takeaways and next steps

## 🎯 Learning Outcomes

After completing all notebooks, you will be able to:

✅ Understand NGLab's hybrid Rust/Python/TypeScript architecture  
✅ Build and train deep RL agents (PPO, SAC, DQN)  
✅ Optimize hyperparameters with evolutionary algorithms  
✅ Deploy trained agents in production environments  
✅ Analyze trading performance with comprehensive backtests  

## 🔧 Troubleshooting

### Rust Module Not Found
```bash
cd ../rust
maturin develop --release
```

### Missing Dependencies
```bash
pip install -r ../python/requirements.txt
```

### GPU Not Detected
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

## 📚 Additional Resources

- [TUTORIAL.md](../TUTORIAL.md) - Comprehensive technical reference
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System design documentation
- [AGENTS.md](../AGENTS.md) - Agent architectures and algorithms

## 🤝 Contributing

Found an issue or want to improve a notebook? See [CONTRIBUTING.md](../git/CONTRIBUTING.md)

---

*Happy Learning! 🚀*
