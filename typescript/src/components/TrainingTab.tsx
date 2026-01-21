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
  Upload,
  FileText,
  CheckCircle,
} from "lucide-react";
import clsx from "clsx";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";

// Model categories with their models
/**
 * Categorized list of available deep learning models and their specific hyperparameters.
 * Organized by architecture type (RNNs, Transformers, CNNs, etc.).
 */
const MODEL_CATEGORIES = {
  // Deep Learning Models
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
    {
      name: "Mamba",
      description: "State-space model with selective scan",
      params: ["hidden_dim", "n_layers", "forecast_horizon"],
    },
  ],
  Transformers: [
    {
      name: "NSTransformer",
      description: "Non-Stationary Transformer for time series",
      params: ["embed_dim", "hidden_dim", "pred_len", "seq_len"],
    },
    {
      name: "Attention",
      description: "Multi-Head Attention Network",
      params: ["d_model", "num_heads", "num_layers", "d_ff"],
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
    {
      name: "Deconv",
      description: "Deconvolutional Network",
      params: ["hidden_channels"],
    },
    {
      name: "AutoDeconv",
      description: "Auto Deconvolutional Network",
      params: ["latent_dim", "hidden_channels"],
    },
    {
      name: "DCIGN",
      description: "Deep Convolutional Inverse Graphics Network",
      params: ["latent_dim", "hidden_channels", "num_intrinsic"],
    },
    {
      name: "Capsule",
      description: "Capsule Network layer",
      params: ["in_caps", "out_caps", "out_dim"],
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
      name: "StackedAE",
      description: "Stacked AutoEncoder",
      params: ["hidden_dims", "latent_dim"],
    },
    {
      name: "VAE",
      description: "Variational AutoEncoder",
      params: ["latent_dim", "d_model", "encoder_type"],
    },
  ],
  "Spiking": [
    {
      name: "SNN",
      description: "Spiking Neural Network with LIF neurons",
      params: ["hidden_dim", "n_layers", "decay", "threshold"],
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
    {
      name: "Hopfield",
      description: "Modern Hopfield Network",
      params: [],
    },
  ],
  "Probabilistic": [
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
    {
      name: "BM",
      description: "Boltzmann Machine",
      params: [],
    },
    {
      name: "MarkovChain",
      description: "Learnable Markov Chain",
      params: ["num_states"],
    },
    {
      name: "Flow",
      description: "Normalizing Flow for density estimation",
      params: ["hidden_dim", "num_layers"],
    },
  ],
  "General Neural Networks": [
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
      name: "PINN",
      description: "Physics-Informed Neural Network",
      params: ["hidden_dim", "num_layers"],
    },
    {
      name: "NODE",
      description: "Neural Ordinary Differential Equation",
      params: ["hidden_dim", "num_layers"],
    },
  ],
  "Competitive Learning": [
    {
      name: "SOM",
      description: "Self-Organizing Map (Kohonen)",
      params: ["grid_size"],
    },
    {
      name: "LVQ",
      description: "Learning Vector Quantization",
      params: ["num_classes", "prototypes_per_class"],
    },
  ],

  // Classical Machine Learning Models
  "Linear Models": [
    {
      name: "LinearRegression",
      description: "Ordinary Least Squares Regression",
      params: [],
    },
    {
      name: "Ridge",
      description: "Ridge Regression (L2 regularization)",
      params: ["alpha"],
    },
    {
      name: "Lasso",
      description: "Lasso Regression (L1 regularization)",
      params: ["alpha"],
    },
    {
      name: "ElasticNet",
      description: "Elastic Net (L1 + L2 regularization)",
      params: ["alpha", "l1_ratio"],
    },
    {
      name: "LARS",
      description: "Least Angle Regression",
      params: ["n_nonzero_coefs"],
    },
    {
      name: "LogisticRegression",
      description: "Logistic Regression for classification",
      params: [],
    },
    {
      name: "Polynomial",
      description: "Polynomial Regression",
      params: ["degree"],
    },
    {
      name: "OLSR",
      description: "Ordinary Least Squares Regression (alias)",
      params: [],
    },
    {
      name: "Stepwise",
      description: "Stepwise Regression with feature selection",
      params: ["direction", "n_features_to_select"],
    },
    {
      name: "MARS",
      description: "Multivariate Adaptive Regression Splines",
      params: ["n_segments"],
    },
    {
      name: "LOESS",
      description: "Locally Estimated Scatterplot Smoothing",
      params: ["frac", "it"],
    },
  ],
  "Decision Trees": [
    {
      name: "DecisionTree",
      description: "Decision Tree (CART)",
      params: ["task", "max_depth"],
    },
    {
      name: "CART",
      description: "Classification and Regression Tree",
      params: ["task", "max_depth"],
    },
    {
      name: "ID3",
      description: "Iterative Dichotomiser 3",
      params: ["task"],
    },
    {
      name: "C45",
      description: "C4.5 Algorithm",
      params: ["task"],
    },
    {
      name: "C50",
      description: "C5.0 Algorithm",
      params: ["task"],
    },
    {
      name: "CHAID",
      description: "Chi-squared Automatic Interaction Detection",
      params: ["task"],
    },
    {
      name: "DecisionStump",
      description: "Decision Stump (depth=1)",
      params: ["task"],
    },
    {
      name: "ConditionalTree",
      description: "Conditional Decision Tree",
      params: ["task", "min_impurity_decrease"],
    },
    {
      name: "M5",
      description: "M5 Model Tree",
      params: [],
    },
    {
      name: "RandomForest",
      description: "Random Forest ensemble",
      params: ["task", "n_estimators"],
    },
  ],
  "Boosting Methods": [
    {
      name: "GradientBoosting",
      description: "Gradient Boosting Machine",
      params: ["task", "n_estimators", "learning_rate"],
    },
    {
      name: "GBRT",
      description: "Gradient Boosted Regression Trees",
      params: ["task", "n_estimators"],
    },
    {
      name: "AdaBoost",
      description: "Adaptive Boosting",
      params: ["task", "n_estimators"],
    },
    {
      name: "XGBoost",
      description: "Extreme Gradient Boosting",
      params: ["task", "n_estimators", "max_depth"],
    },
    {
      name: "LightGBM",
      description: "Light Gradient Boosting Machine",
      params: ["task", "n_estimators", "num_leaves"],
    },
  ],
  "Ensemble Methods": [
    {
      name: "Bagging",
      description: "Bootstrap Aggregating",
      params: ["task", "n_estimators"],
    },
    {
      name: "Stacking",
      description: "Stacked Generalization",
      params: ["task"],
    },
    {
      name: "Voting",
      description: "Voting Ensemble",
      params: ["task"],
    },
    {
      name: "WeightedAverage",
      description: "Weighted Average (Blending)",
      params: ["task"],
    },
  ],
  "Support Vector Machines": [
    {
      name: "SVM",
      description: "Support Vector Machine",
      params: ["task", "kernel"],
    },
    {
      name: "SVR",
      description: "Support Vector Regression",
      params: ["kernel"],
    },
    {
      name: "LinearSVM",
      description: "Linear SVM",
      params: ["task"],
    },
    {
      name: "NuSVM",
      description: "Nu-Support Vector Machine",
      params: ["task", "nu"],
    },
    {
      name: "OneClassSVM",
      description: "One-Class SVM (anomaly detection)",
      params: ["nu"],
    },
    {
      name: "LSSVM",
      description: "Least-Squares SVM",
      params: ["alpha", "kernel"],
    },
    {
      name: "TWSVM",
      description: "Twin Support Vector Machine",
      params: ["c1", "c2"],
    },
  ],
  "Naive Bayes": [
    {
      name: "NaiveBayes",
      description: "Naive Bayes Classifier",
      params: ["type"],
    },
    {
      name: "GaussianNB",
      description: "Gaussian Naive Bayes",
      params: [],
    },
    {
      name: "MultinomialNB",
      description: "Multinomial Naive Bayes",
      params: [],
    },
    {
      name: "AODE",
      description: "Averaged One-Dependence Estimators",
      params: ["n_estimators"],
    },
    {
      name: "BayesianNetwork",
      description: "Bayesian Belief Network",
      params: ["structure"],
    },
  ],
  "Nearest Neighbors": [
    {
      name: "kNN",
      description: "k-Nearest Neighbors",
      params: ["task", "n_neighbors"],
    },
    {
      name: "LWL",
      description: "Locally Weighted Learning",
      params: ["task", "n_neighbors", "kernel"],
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

  // CSV file state
  const [csvPath, setCsvPath] = useState<string | null>(null);
  const [csvColumns, setCsvColumns] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState<string | null>(null);
  const [loadingColumns, setLoadingColumns] = useState(false);
  const [trainLoss, setTrainLoss] = useState<number | null>(null);
  const [valLoss, setValLoss] = useState<number | null>(null);
  const [currentEpoch, setCurrentEpoch] = useState(0);

  /**
   * Toggles the visibility of a model category in the sidebar.
   * @param category - The category name to toggle.
   */
  const toggleCategory = (category: string) => {
    const newExpanded = new Set(expandedCategories);
    if (newExpanded.has(category)) {
      newExpanded.delete(category);
    } else {
      newExpanded.add(category);
    }
    setExpandedCategories(newExpanded);
  };

  /**
   * Opens a file dialog to select a CSV dataset.
   * Automatically attempts to detect and list available columns upon selection.
   */
  const handleSelectCsv = async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [{ name: "CSV", extensions: ["csv"] }],
      });
      if (selected && typeof selected === "string") {
        setCsvPath(selected);
        setTargetColumn(null);
        setCsvColumns([]);
        setLoadingColumns(true);

        try {
          const columns = await invoke<string[]>("list_csv_columns", {
            csvPath: selected,
          });
          setCsvColumns(columns);
          if (columns.length > 0) {
            setTargetColumn(columns[0]);
          }
        } catch (err) {
          console.error("Failed to load columns:", err);
          setTrainingLogs((logs) => [
            ...logs,
            `[${new Date().toLocaleTimeString()}] Error loading CSV columns: ${err}`,
          ]);
        } finally {
          setLoadingColumns(false);
        }
      }
    } catch (err) {
      console.error("Failed to open file dialog:", err);
    }
  };

  /**
   * Initiates the model training process on the Rust backend.
   * Sets up real-time event listeners for progress updates (loss, epoch).
   */
  const handleStartTraining = async () => {
    if (!selectedModel || !csvPath || !targetColumn) return;
    setIsTraining(true);
    setTrainingProgress(0);
    setCurrentEpoch(0);
    setTrainLoss(null);
    setValLoss(null);
    setTrainingLogs([
      `[${new Date().toLocaleTimeString()}] Starting training with ${selectedModel}...`,
      `[${new Date().toLocaleTimeString()}] CSV: ${csvPath}`,
      `[${new Date().toLocaleTimeString()}] Target: ${targetColumn}`,
    ]);

    // Listen for training progress events
    let unlisten: UnlistenFn | null = null;
    try {
      unlisten = await listen<{
        type: string;
        epoch?: number;
        total_epochs?: number;
        train_loss?: number;
        val_loss?: number;
        percent?: number;
        model_path?: string;
        message?: string;
      }>("training-progress", (event) => {
        const data = event.payload;
        if (data.type === "progress") {
          setTrainingProgress(data.percent || 0);
          setCurrentEpoch(data.epoch || 0);
          setTrainLoss(data.train_loss || null);
          setValLoss(data.val_loss || null);
          if (data.epoch && data.epoch % 10 === 0) {
            setTrainingLogs((logs) =>
              [
                ...logs,
                `[${new Date().toLocaleTimeString()}] Epoch ${data.epoch}/${data.total_epochs}: train_loss=${data.train_loss?.toFixed(6)}, val_loss=${data.val_loss?.toFixed(6) || "N/A"}`,
              ].slice(-30)
            );
          }
        }
      });

      // Call training command
      const modelPath = await invoke<string>("train_model", {
        csvPath,
        targetColumn,
        modelName: selectedModel,
        epochs: config.epochs,
        batchSize: config.batchSize,
        learningRate: config.learningRate,
        seqLen: config.seqLen,
        predLen: config.predLen,
        trainSplit: config.trainSplit,
        modelParams,
      });

      setTrainingLogs((logs) => [
        ...logs,
        `[${new Date().toLocaleTimeString()}] Training completed!`,
        `[${new Date().toLocaleTimeString()}] Model saved to: ${modelPath}`,
      ]);
      setTrainingProgress(100);
    } catch (err) {
      console.error("Training failed:", err);
      setTrainingLogs((logs) => [
        ...logs,
        `[${new Date().toLocaleTimeString()}] Training failed: ${err}`,
      ]);
    } finally {
      if (unlisten) {
        unlisten();
      }
      setIsTraining(false);
    }
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
                      disabled={!csvPath || !targetColumn}
                      className={clsx(
                        "flex items-center gap-2 px-4 py-2 rounded-lg transition-colors font-medium text-sm",
                        csvPath && targetColumn
                          ? "bg-indigo-600 hover:bg-indigo-500"
                          : "bg-slate-700 cursor-not-allowed opacity-50"
                      )}
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
                  <div className="space-y-4">
                    {/* CSV File Selection */}
                    <div className="flex items-center gap-4">
                      <button
                        onClick={handleSelectCsv}
                        className="flex items-center gap-2 px-4 py-2 bg-indigo-500/20 border border-indigo-500/50 rounded-lg hover:bg-indigo-500/30 transition-colors"
                      >
                        <Upload size={16} />
                        <span className="text-sm font-medium">
                          Select CSV File
                        </span>
                      </button>
                      {csvPath && (
                        <div className="flex items-center gap-2 text-sm text-slate-400">
                          <FileText size={16} className="text-emerald-400" />
                          <span className="truncate max-w-xs">
                            {csvPath.split("/").pop()}
                          </span>
                          <CheckCircle size={14} className="text-emerald-400" />
                        </div>
                      )}
                    </div>

                    {/* Column Selection */}
                    {csvColumns.length > 0 && (
                      <div>
                        <label className="text-xs text-slate-500 uppercase">
                          Target Column
                        </label>
                        <select
                          value={targetColumn || ""}
                          onChange={(e) => setTargetColumn(e.target.value)}
                          className="w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                        >
                          {csvColumns.map((col) => (
                            <option key={col} value={col}>
                              {col}
                            </option>
                          ))}
                        </select>
                        <p className="text-xs text-slate-500 mt-1">
                          Select the column containing values to predict
                        </p>
                      </div>
                    )}

                    {loadingColumns && (
                      <div className="text-sm text-slate-400 animate-pulse">
                        Loading columns...
                      </div>
                    )}

                    {!csvPath && (
                      <p className="text-sm text-slate-500">
                        Upload a CSV file containing time series data for
                        training.
                      </p>
                    )}
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
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                <div className="text-lg font-mono font-bold text-indigo-400">
                  {currentEpoch}/{config.epochs}
                </div>
                <div className="text-[10px] text-slate-500 uppercase">
                  Epoch
                </div>
              </div>
              <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                <div className="text-lg font-mono font-bold text-blue-400">
                  {trainLoss !== null ? trainLoss.toFixed(4) : "--"}
                </div>
                <div className="text-[10px] text-slate-500 uppercase">
                  Train Loss
                </div>
              </div>
              <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                <div className="text-lg font-mono font-bold text-emerald-400">
                  {valLoss !== null ? valLoss.toFixed(4) : "--"}
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
