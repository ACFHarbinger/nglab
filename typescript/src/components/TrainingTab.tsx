/**
 * @module components/TrainingTab
 * @description Comprehensive deep learning training interface with support for multiple architectures.
 */
import { useState } from "react";
import {
  Play,
  Square,
  Settings,
  Cpu,
  Database,
  TrendingUp,
  Zap,
  Clock,
  BarChart3,
  ChevronDown,
  ChevronRight,
  Info,
} from "lucide-react";
import clsx from "clsx";

// Model categories with their models
const MODEL_CATEGORIES = {
  "Recurrent Networks": [
    {
      name: "LSTM",
      description: "Long Short-Term Memory network",
      params: ["hidden_dim", "n_layers", "dropout"],
    },
    {
      name: "GRU",
      description: "Gated Recurrent Unit network",
      params: ["hidden_dim", "n_layers", "dropout"],
    },
    {
      name: "xLSTM",
      description: "Extended LSTM with sLSTM/mLSTM cells",
      params: ["hidden_dim", "n_layers", "cell_type", "num_heads"],
    },
  ],
  Transformers: [
    {
      name: "NSTransformer",
      description: "Non-Stationary Transformer for time series",
      params: ["embed_dim", "hidden_dim", "pred_len", "seq_len"],
    },
    {
      name: "AttentionNet",
      description: "Multi-Head Attention Network",
      params: ["hidden_dim", "num_heads", "num_layers"],
    },
  ],
  Convolutional: [
    {
      name: "CNN",
      description: "Rolling Window CNN for sequences",
      params: ["hidden_dim", "seq_len"],
    },
    {
      name: "DCN",
      description: "Deep Convolutional Network",
      params: ["hidden_channels"],
    },
    {
      name: "ResNet",
      description: "Deep Residual Network",
      params: ["hidden_dim", "num_blocks", "dropout"],
    },
  ],
  Autoencoders: [
    {
      name: "AE",
      description: "Standard AutoEncoder",
      params: ["hidden_dims", "latent_dim"],
    },
    {
      name: "DAE",
      description: "Denoising AutoEncoder",
      params: ["hidden_dims", "latent_dim", "noise_std"],
    },
    {
      name: "SAE",
      description: "Sparse AutoEncoder",
      params: ["hidden_dims", "latent_dim", "sparsity_target"],
    },
    {
      name: "VAE",
      description: "Variational AutoEncoder",
      params: ["hidden_dims", "latent_dim"],
    },
  ],
  "Spiking & Reservoir": [
    {
      name: "SNN",
      description: "Spiking Neural Network with LIF neurons",
      params: ["hidden_dim", "n_layers", "decay", "threshold"],
    },
    {
      name: "ESN",
      description: "Echo State Network (reservoir computing)",
      params: ["reservoir_dim", "spectral_radius", "sparsity"],
    },
    {
      name: "LSM",
      description: "Liquid State Machine",
      params: ["liquid_size", "connection_prob", "spectral_radius"],
    },
  ],
  "Memory Networks": [
    {
      name: "NTM",
      description: "Neural Turing Machine",
      params: ["hidden_dim", "memory_size", "memory_dim"],
    },
    {
      name: "DNC",
      description: "Differentiable Neural Computer",
      params: ["hidden_dim", "memory_size", "memory_dim", "num_reads"],
    },
    { name: "Hopfield", description: "Modern Hopfield Network", params: [] },
  ],
  Probabilistic: [
    {
      name: "RBM",
      description: "Restricted Boltzmann Machine",
      params: ["hidden_dim"],
    },
    {
      name: "DBN",
      description: "Deep Belief Network",
      params: ["hidden_dims"],
    },
    { name: "BM", description: "Boltzmann Machine", params: [] },
    {
      name: "MarkovChain",
      description: "Learnable Markov Chain",
      params: ["num_states"],
    },
  ],
  Classical: [
    {
      name: "MLP",
      description: "Multi-Layer Perceptron",
      params: ["hidden_dims", "dropout", "activation"],
    },
    {
      name: "RBF",
      description: "Radial Basis Function Network",
      params: ["num_centers", "sigma"],
    },
    {
      name: "ELM",
      description: "Extreme Learning Machine",
      params: ["hidden_dim", "activation"],
    },
    {
      name: "Perceptron",
      description: "Single-layer Perceptron",
      params: ["activation"],
    },
    {
      name: "SOM",
      description: "Self-Organizing Map (Kohonen)",
      params: ["grid_size"],
    },
  ],
  Advanced: [
    {
      name: "Mamba",
      description: "State-space model with selective scan",
      params: ["hidden_dim", "n_layers", "forecast_horizon"],
    },
    {
      name: "Capsule",
      description: "Capsule Network layer",
      params: ["in_caps", "out_caps", "out_dim"],
    },
    {
      name: "NormalizingFlow",
      description: "Normalizing Flow for density estimation",
      params: ["hidden_dim", "num_flows"],
    },
    {
      name: "NeuralODE",
      description: "Neural Ordinary Differential Equation",
      params: ["hidden_dim"],
    },
    {
      name: "PINN",
      description: "Physics-Informed Neural Network",
      params: ["hidden_dims"],
    },
    {
      name: "DCIGN",
      description: "Deep Convolutional Inverse Graphics Network",
      params: ["latent_dim", "hidden_channels"],
    },
  ],
};

/**
 * Configuration for the training process.
 */
interface TrainingConfig {
  /** Number of training epochs. */
  epochs: number;
  /** Size of each training batch. */
  batchSize: number;
  /** Rate at which the model learns. */
  learningRate: number;
  /** Length of the input sequence context. */
  seqLen: number;
  /** Number of future steps to predict. */
  predLen: number;
  /** Ratio of data used for training vs validation (0.0 to 1.0). */
  trainSplit: number;
}

/**
 * Main component for configuring and running deep learning model training.
 *
 * Provides a categorized list of architectures (RNN, Transformers, CNN, etc.)
 * and exposes hyperparameters for tuning.
 */
export default function TrainingTab() {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(["Recurrent Networks", "Transformers"]),
  );
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);

  const [config, setConfig] = useState<TrainingConfig>({
    epochs: 100,
    batchSize: 32,
    learningRate: 0.001,
    seqLen: 30,
    predLen: 1,
    trainSplit: 0.8,
  });

  const [modelParams, setModelParams] = useState<
    Record<string, number | string>
  >({
    hidden_dim: 128,
    n_layers: 2,
    dropout: 0.1,
    embed_dim: 64,
    num_heads: 4,
  });

  const toggleCategory = (category: string) => {
    const newExpanded = new Set(expandedCategories);
    if (newExpanded.has(category)) {
      newExpanded.delete(category);
    } else {
      newExpanded.add(category);
    }
    setExpandedCategories(newExpanded);
  };

  const handleStartTraining = () => {
    if (!selectedModel) return;
    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingLogs([
      `[${new Date().toLocaleTimeString()}] Starting training with ${selectedModel}...`,
    ]);

    // Simulate training progress
    const interval = setInterval(() => {
      setTrainingProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          setTrainingLogs((logs) => [
            ...logs,
            `[${new Date().toLocaleTimeString()}] Training completed!`,
          ]);
          return 100;
        }
        const newProgress = prev + Math.random() * 5;
        if (Math.random() > 0.7) {
          setTrainingLogs((logs) =>
            [
              ...logs,
              `[${new Date().toLocaleTimeString()}] Epoch ${Math.floor(newProgress)}: loss=0.${Math.floor(
                Math.random() * 1000,
              )
                .toString()
                .padStart(4, "0")}`,
            ].slice(-20),
          );
        }
        return Math.min(newProgress, 100);
      });
    }, 200);
  };

  const handleStopTraining = () => {
    setIsTraining(false);
    setTrainingLogs((logs) => [
      ...logs,
      `[${new Date().toLocaleTimeString()}] Training stopped by user.`,
    ]);
  };

  const getModelInfo = (modelName: string) => {
    for (const category of Object.values(MODEL_CATEGORIES)) {
      const model = category.find((m) => m.name === modelName);
      if (model) return model;
    }
    return null;
  };

  const selectedModelInfo = selectedModel ? getModelInfo(selectedModel) : null;

  return (
    <div className="flex h-full bg-slate-950 overflow-hidden">
      {/* Left Panel: Model Selection */}
      <div className="w-80 border-r border-slate-800 flex flex-col">
        <div className="p-4 border-b border-slate-800">
          <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-400">
            <Cpu size={16} /> Select Model
          </div>
        </div>
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {Object.entries(MODEL_CATEGORIES).map(([category, models]) => (
            <div key={category} className="border-b border-slate-800/50">
              <button
                onClick={() => toggleCategory(category)}
                className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-slate-300 hover:bg-slate-800/50 transition-colors"
              >
                <span>{category}</span>
                {expandedCategories.has(category) ? (
                  <ChevronDown size={16} className="text-slate-500" />
                ) : (
                  <ChevronRight size={16} className="text-slate-500" />
                )}
              </button>
              {expandedCategories.has(category) && (
                <div className="pb-2">
                  {models.map((model) => (
                    <button
                      key={model.name}
                      onClick={() => setSelectedModel(model.name)}
                      className={clsx(
                        "w-full flex items-center gap-3 px-4 py-2 text-left transition-colors",
                        selectedModel === model.name
                          ? "bg-indigo-500/20 border-l-2 border-indigo-500 text-white"
                          : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200 border-l-2 border-transparent",
                      )}
                    >
                      <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center text-xs font-mono text-slate-500">
                        {model.name.slice(0, 2).toUpperCase()}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium truncate">
                          {model.name}
                        </div>
                        <div className="text-[10px] text-slate-500 truncate">
                          {model.description}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Center Panel: Configuration */}
      <div className="flex-1 flex flex-col min-w-0 border-r border-slate-800">
        {selectedModel ? (
          <>
            {/* Model Header */}
            <div className="p-6 border-b border-slate-800">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-bold text-white flex items-center gap-2">
                    <Zap size={20} className="text-indigo-400" />
                    {selectedModel}
                  </h2>
                  <p className="text-sm text-slate-400 mt-1">
                    {selectedModelInfo?.description}
                  </p>
                </div>
                <div className="flex gap-2">
                  {!isTraining ? (
                    <button
                      onClick={handleStartTraining}
                      className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-4 py-2 rounded-lg transition-colors font-medium text-sm"
                    >
                      <Play size={16} fill="currentColor" /> Start Training
                    </button>
                  ) : (
                    <button
                      onClick={handleStopTraining}
                      className="flex items-center gap-2 bg-rose-600 hover:bg-rose-500 px-4 py-2 rounded-lg transition-colors font-medium text-sm"
                    >
                      <Square size={16} fill="currentColor" /> Stop
                    </button>
                  )}
                </div>
              </div>

              {/* Progress Bar */}
              {isTraining && (
                <div className="mt-4">
                  <div className="flex justify-between text-xs text-slate-400 mb-1">
                    <span>Training Progress</span>
                    <span>{Math.floor(trainingProgress)}%</span>
                  </div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-200"
                      style={{ width: `${trainingProgress}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Configuration Sections */}
            <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
              <div className="grid grid-cols-2 gap-6">
                {/* Training Config */}
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
                    <Settings size={14} /> Training Configuration
                  </h3>
                  <div className="space-y-4">
                    <div>
                      <label className="text-xs text-slate-500 uppercase">
                        Epochs
                      </label>
                      <input
                        type="number"
                        value={config.epochs}
                        onChange={(e) =>
                          setConfig({
                            ...config,
                            epochs: parseInt(e.target.value) || 0,
                          })
                        }
                        className="w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-slate-500 uppercase">
                        Batch Size
                      </label>
                      <input
                        type="number"
                        value={config.batchSize}
                        onChange={(e) =>
                          setConfig({
                            ...config,
                            batchSize: parseInt(e.target.value) || 0,
                          })
                        }
                        className="w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-slate-500 uppercase">
                        Learning Rate
                      </label>
                      <input
                        type="number"
                        step="0.0001"
                        value={config.learningRate}
                        onChange={(e) =>
                          setConfig({
                            ...config,
                            learningRate: parseFloat(e.target.value) || 0,
                          })
                        }
                        className="w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-slate-500 uppercase">
                        Train/Val Split
                      </label>
                      <input
                        type="number"
                        step="0.05"
                        min="0.5"
                        max="0.95"
                        value={config.trainSplit}
                        onChange={(e) =>
                          setConfig({
                            ...config,
                            trainSplit: parseFloat(e.target.value) || 0.8,
                          })
                        }
                        className="w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                      />
                    </div>
                  </div>
                </div>

                {/* Sequence Config */}
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
                    <TrendingUp size={14} /> Sequence Configuration
                  </h3>
                  <div className="space-y-4">
                    <div>
                      <label className="text-xs text-slate-500 uppercase">
                        Sequence Length (Input)
                      </label>
                      <input
                        type="number"
                        value={config.seqLen}
                        onChange={(e) =>
                          setConfig({
                            ...config,
                            seqLen: parseInt(e.target.value) || 0,
                          })
                        }
                        className="w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-slate-500 uppercase">
                        Prediction Length (Output)
                      </label>
                      <input
                        type="number"
                        value={config.predLen}
                        onChange={(e) =>
                          setConfig({
                            ...config,
                            predLen: parseInt(e.target.value) || 0,
                          })
                        }
                        className="w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                      />
                    </div>
                    <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-700/50">
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <Info size={12} />
                        <span>
                          Uses {config.seqLen} timesteps to predict{" "}
                          {config.predLen} future values
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Model-Specific Params */}
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 col-span-2">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
                    <Cpu size={14} /> Model Parameters ({selectedModel})
                  </h3>
                  <div className="grid grid-cols-4 gap-4">
                    {selectedModelInfo?.params.map((param) => (
                      <div key={param}>
                        <label className="text-xs text-slate-500 uppercase">
                          {param.replace(/_/g, " ")}
                        </label>
                        <input
                          type={
                            param.includes("dropout") ||
                            param.includes("rate") ||
                            param.includes("prob")
                              ? "number"
                              : "text"
                          }
                          step={
                            param.includes("dropout") ||
                            param.includes("rate") ||
                            param.includes("prob")
                              ? "0.01"
                              : undefined
                          }
                          value={modelParams[param] ?? ""}
                          onChange={(e) =>
                            setModelParams({
                              ...modelParams,
                              [param]: e.target.value,
                            })
                          }
                          className="w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                          placeholder={`Enter ${param}`}
                        />
                      </div>
                    ))}
                    {selectedModelInfo?.params.length === 0 && (
                      <div className="col-span-4 text-sm text-slate-500 italic">
                        This model has no configurable parameters.
                      </div>
                    )}
                  </div>
                </div>

                {/* Data Source */}
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 col-span-2">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
                    <Database size={14} /> Data Source
                  </h3>
                  <div className="grid grid-cols-3 gap-4">
                    <button className="p-4 bg-indigo-500/20 border border-indigo-500/50 rounded-lg text-left hover:bg-indigo-500/30 transition-colors">
                      <div className="text-sm font-medium text-white">
                        Polymarket Live
                      </div>
                      <div className="text-xs text-slate-400 mt-1">
                        Real-time market data
                      </div>
                    </button>
                    <button className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg text-left hover:bg-slate-800 transition-colors">
                      <div className="text-sm font-medium text-slate-300">
                        Historical CSV
                      </div>
                      <div className="text-xs text-slate-500 mt-1">
                        Import from file
                      </div>
                    </button>
                    <button className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg text-left hover:bg-slate-800 transition-colors">
                      <div className="text-sm font-medium text-slate-300">
                        Synthetic
                      </div>
                      <div className="text-xs text-slate-500 mt-1">
                        Generated test data
                      </div>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <Cpu size={48} className="text-slate-700 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-slate-400">
                Select a Model
              </h3>
              <p className="text-sm text-slate-600 mt-1">
                Choose a model from the left panel to configure training
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Right Panel: Logs & Metrics */}
      <div className="w-80 flex flex-col">
        <div className="p-4 border-b border-slate-800">
          <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-400">
            <BarChart3 size={16} /> Training Logs
          </div>
        </div>
        <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
          {trainingLogs.length > 0 ? (
            <div className="space-y-1 font-mono text-xs">
              {trainingLogs.map((log, i) => (
                <div
                  key={i}
                  className="text-slate-400 border-l-2 border-slate-700 pl-2 py-0.5"
                >
                  {log}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-slate-600 text-sm py-8">
              <Clock size={24} className="mx-auto mb-2 text-slate-700" />
              Training logs will appear here
            </div>
          )}
        </div>

        {/* Quick Stats */}
        {isTraining && (
          <div className="border-t border-slate-800 p-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                <div className="text-lg font-mono font-bold text-indigo-400">
                  {Math.floor(trainingProgress)}
                </div>
                <div className="text-[10px] text-slate-500 uppercase">
                  Epoch
                </div>
              </div>
              <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                <div className="text-lg font-mono font-bold text-emerald-400">
                  0.
                  {Math.floor(Math.random() * 100)
                    .toString()
                    .padStart(2, "0")}
                </div>
                <div className="text-[10px] text-slate-500 uppercase">
                  Val Loss
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
