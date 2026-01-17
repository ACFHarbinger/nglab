"""
Unified Backbone for Time Series Models.
"""

from torch import nn

from .deep import *
from .mac import *


class TimeSeriesBackbone(nn.Module):
    """
    Unified Backbone for Time Series.
    Wraps specific implementations (Transformer, LSTM, etc).
    """

    def __init__(self, cfg):
        """
        Initialize TimeSeriesBackbone.

        Args:
            cfg: Configuration dictionary defining the model architecture.
        """
        super().__init__()
        self.cfg = cfg
        model_name = cfg.get("name", "NSTransformer")

        if model_name == "NSTransformer":
            self.model = NSTransformer(
                pred_len=cfg.get("pred_len", 1),
                seq_len=cfg.get("seq_len", 30),
                input_dim=cfg.get("feature_dim", 12),
                embed_dim=cfg.get("embed_dim", 64),
                hidden_dim=cfg.get("hidden_dim", 128),
                output_dim=cfg.get("output_dim", 64),
                learner_dims=cfg.get("learner_dims", [64]),
            )
        elif model_name == "Mamba":
            self.model = TSMamba(
                input_dim=cfg.get("feature_dim", 12),
                output_dim=1,
                d_model=cfg.get("hidden_dim", 128),
                n_layers=cfg.get("num_layers", 2),
                forecast_horizon=cfg.get("pred_len", 1),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "LSTM":
            self.model = LSTM(
                input_dim=cfg.get("feature_dim", 12),
                output_dim=cfg.get("output_dim", 1),
                hidden_dim=cfg.get("hidden_dim", 128),
                n_layers=cfg.get("num_layers", 2),
                dropout=cfg.get("dropout", 0.0),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "GRU":
            self.model = GRU(
                input_dim=cfg.get("feature_dim", 12),
                output_dim=cfg.get("output_dim", 1),
                hidden_dim=cfg.get("hidden_dim", 128),
                n_layers=cfg.get("num_layers", 2),
                dropout=cfg.get("dropout", 0.0),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "xLSTM":
            self.model = xLSTM(
                input_dim=cfg.get("feature_dim", 12),
                output_dim=1,
                hidden_dim=cfg.get("hidden_dim", 128),
                n_layers=cfg.get("num_layers", 2),
                dropout=cfg.get("dropout", 0.0),
                output_type=cfg.get("output_type", "embedding"),
                cell_type=cfg.get("cell_type", "slstm"),
                num_heads=cfg.get("num_heads", 4),
            )
        elif model_name == "SNN":
            self.model = SNN(
                input_dim=cfg.get("feature_dim", 12),
                output_dim=cfg.get("output_dim", 1),
                hidden_dim=cfg.get("hidden_dim", 128),
                n_layers=cfg.get("num_layers", 2),
                dropout=cfg.get("dropout", 0.0),
                output_type=cfg.get("output_type", "embedding"),
                decay=cfg.get("decay", 0.9),
                threshold=cfg.get("threshold", 1.0),
            )
        elif model_name == "MLP":
            self.model = MLP(
                input_dim=cfg.get("feature_dim", 12),
                hidden_dims=cfg.get("hidden_dims", [128, 64]),
                output_dim=cfg.get("output_dim", 1),
                dropout=cfg.get("dropout", 0.0),
                activation=cfg.get("activation", "relu"),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "RBF":
            self.model = RBF(
                input_dim=cfg.get("feature_dim", 12),
                num_centers=cfg.get("hidden_dim", 100),
                output_dim=cfg.get("output_dim", 1),
                sigma=cfg.get("sigma", 1.0),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "AE":
            self.model = AutoEncoder(
                input_dim=cfg.get("feature_dim", 12),
                hidden_dims=cfg.get("hidden_dims", [64]),
                latent_dim=cfg.get("hidden_dim", 32),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "DAE":
            self.model = DenoisingAE(
                input_dim=cfg.get("feature_dim", 12),
                hidden_dims=cfg.get("hidden_dims", [64]),
                latent_dim=cfg.get("hidden_dim", 32),
                noise_std=cfg.get("noise_std", 0.1),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "SAE":
            self.model = SparseAE(
                input_dim=cfg.get("feature_dim", 12),
                hidden_dims=cfg.get("hidden_dims", [64]),
                latent_dim=cfg.get("hidden_dim", 32),
                sparsity_target=cfg.get("sparsity_target", 0.05),
                sparsity_weight=cfg.get("sparsity_weight", 0.1),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "StackedAE":
            self.model = StackedAutoEncoder(
                layer_sizes=[cfg.get("feature_dim", 12)]
                + cfg.get("hidden_dims", [64, 32])
                + [cfg.get("latent_dim", 16)],
                 output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "Hopfield":
            self.model = HopfieldNetwork(
                size=cfg.get("feature_dim", 12),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "RBM":
            self.model = RBM(
                visible_dim=cfg.get("feature_dim", 12),
                hidden_dim=cfg.get("hidden_dim", 64),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "ESN":
            self.model = EchoStateNetwork(
                input_dim=cfg.get("feature_dim", 12),
                reservoir_dim=cfg.get("hidden_dim", 500),
                output_dim=cfg.get("output_dim", 1),
                spectral_radius=cfg.get("spectral_radius", 0.9),
                sparsity=cfg.get("sparsity", 0.1),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "ELM":
            self.model = ELM(
                input_dim=cfg.get("feature_dim", 12),
                hidden_dim=cfg.get("hidden_dim", 500),
                output_dim=cfg.get("output_dim", 1),
                activation=cfg.get("activation", "sigmoid"),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "SOM":
            self.model = KohonenMap(
                input_dim=cfg.get("feature_dim", 12),
                grid_size=cfg.get("grid_size", (10, 10)),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "Capsule":
            self.model = CapsuleLayer(
                in_caps=cfg.get("in_caps", 8),
                in_dim=cfg.get("in_dim", 16),
                out_caps=cfg.get("out_caps", 4),
                out_dim=cfg.get("out_dim", 32),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "CNN":
            self.model = RollingWindowCNN(
                input_dim=cfg.get("feature_dim", 12),
                output_dim=1,
                seq_len=cfg.get("seq_len", 30),
                hidden_dim=cfg.get("hidden_dim", 128),
                output_type=cfg.get("output_type", "embedding"),
            )
        elif model_name == "Perceptron":
            self.model = Perceptron(
                input_dim=cfg.get("feature_dim", 12),
                output_dim=cfg.get("output_dim", 1),
                activation=cfg.get("activation", "sigmoid"),
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "MarkovChain":
            self.model = MarkovChain(
                num_states=cfg.get("num_states", 10),
                output_type=cfg.get("output_type", "prediction"),
                learnable=cfg.get("learnable", True),
            )
        elif model_name == "BM":
            self.model = BoltzmannMachine(
                num_units=cfg.get("feature_dim", 12),
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "DBN":
            self.model = DeepBeliefNetwork(
                layer_sizes=[cfg.get("feature_dim", 12)]
                + cfg.get("hidden_dims", [64, 32]),
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "DCN":
            self.model = DeepConvNet(
                input_dim=cfg.get("feature_dim", 12),
                hidden_channels=cfg.get("hidden_channels", [32, 64, 128]),
                output_dim=cfg.get("output_dim", 1),
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "Deconv":
            self.model = DeconvNet(
                input_dim=cfg.get("feature_dim", 12),
                hidden_channels=cfg.get("hidden_channels", [128, 64, 32]),
                output_dim=cfg.get("output_dim", 1),
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "AutoDeconv":
            self.model = AutoDeconvNet(
                input_dim=cfg.get("feature_dim", 12),
                latent_dim=cfg.get("latent_dim", 64),
                hidden_channels=cfg.get("hidden_channels", [32, 64, 128]),
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "DCIGN":
            latent_dim = cfg.get("latent_dim", 128)
            num_intrinsic = cfg.get("num_intrinsic", latent_dim // 4)
            num_extrinsic = latent_dim - num_intrinsic
            self.model = DCIGN(
                input_dim=cfg.get("feature_dim", 12),
                latent_dim=latent_dim,
                hidden_channels=cfg.get("hidden_channels", [32, 64, 128, 256]),
                num_intrinsic=num_intrinsic,
                num_extrinsic=num_extrinsic,
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "LSM":
            self.model = LiquidStateMachine(
                input_dim=cfg.get("feature_dim", 12),
                liquid_size=cfg.get("liquid_size", 200),
                output_dim=cfg.get("output_dim", 1),
                connection_prob=cfg.get("connection_prob", 0.3),
                spectral_radius=cfg.get("spectral_radius", 1.2),
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "ResNet":
            self.model = DeepResNet(
                input_dim=cfg.get("feature_dim", 12),
                hidden_dim=cfg.get("hidden_dim", 128),
                num_blocks=cfg.get("num_blocks", 4),
                output_dim=cfg.get("output_dim", 1),
                use_conv=cfg.get("use_conv", False),
                dropout=cfg.get("dropout", 0.1),
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "DNC":
            self.model = DNC(
                input_dim=cfg.get("feature_dim", 12),
                hidden_dim=cfg.get("hidden_dim", 128),
                memory_size=cfg.get("memory_size", 64),
                memory_dim=cfg.get("memory_dim", 32),
                num_reads=cfg.get("num_reads", 4),
                output_dim=cfg.get("output_dim", 1),
                controller_type=cfg.get("controller_type", "lstm"),
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "NTM":
            self.model = NTM(
                input_dim=cfg.get("feature_dim", 12),
                hidden_dim=cfg.get("hidden_dim", 128),
                memory_size=cfg.get("memory_size", 128),
                memory_dim=cfg.get("memory_dim", 20),
                num_reads=cfg.get("num_reads", 1),
                num_writes=cfg.get("num_writes", 1),
                output_dim=cfg.get("output_dim", 1),
                controller_type=cfg.get("controller_type", "lstm"),
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "Attention":
            self.model = AttentionNetwork(
                input_dim=cfg.get("feature_dim", 12),
                d_model=cfg.get("d_model", 128),
                num_layers=cfg.get("num_layers", 4),
                num_heads=cfg.get("num_heads", 8),
                d_ff=cfg.get("d_ff", 512),
                output_dim=cfg.get("output_dim", 1),
                dropout=cfg.get("dropout", 0.1),
                max_seq_len=cfg.get("max_seq_len", 1000),
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "Flow":
            self.model = NormalizingFlow(
                input_dim=cfg.get("feature_dim", 12),
                num_layers=cfg.get("num_layers", 4),
                hidden_dim=cfg.get("hidden_dim", 64),
                seq_len=cfg.get("seq_len", 1),
            )
        elif model_name == "NODE":
            self.model = NeuralODE(
                input_dim=cfg.get("feature_dim", 12),
                hidden_dim=cfg.get("hidden_dim", 64),
                output_dim=cfg.get("output_dim", 1),
                num_layers=cfg.get("num_layers", 2),
                time_steps=cfg.get("seq_len", 10),
                horizon=cfg.get("horizon", 1.0),
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "LVQ":
            self.model = LVQ(
                input_dim=cfg.get("feature_dim", 12),
                num_classes=cfg.get("num_classes", 2),
                prototypes_per_class=cfg.get("prototypes_per_class", 1),
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "PINN":
            self.model = PINN(
                input_dim=cfg.get("feature_dim", 2),  # e.g. (x, t)
                hidden_dim=cfg.get("hidden_dim", 20),
                output_dim=cfg.get("output_dim", 1),
                num_layers=cfg.get("num_layers", 4),
                activation=cfg.get("activation", "tanh"),
                output_type=cfg.get("output_type", "prediction"),
            )
        elif model_name == "LinearRegression":
            self.model = LinearRegressionModel(**cfg.get("model_kwargs", {}))
        elif model_name == "Ridge":
            self.model = RidgeRegressionModel(
                alpha=cfg.get("alpha", 1.0), **cfg.get("model_kwargs", {})
            )
        elif model_name == "Lasso":
            self.model = LassoRegressionModel(
                alpha=cfg.get("alpha", 1.0), **cfg.get("model_kwargs", {})
            )
        elif model_name == "LARS":
            self.model = LARSModel(
                 n_nonzero_coefs=cfg.get("n_nonzero_coefs", 500), **cfg.get("model_kwargs", {})
            )
        elif model_name == "ElasticNet":
            self.model = ElasticNetModel(
                alpha=cfg.get("alpha", 1.0),
                l1_ratio=cfg.get("l1_ratio", 0.5),
                **cfg.get("model_kwargs", {}),
            )
        elif model_name == "LogisticRegression":
            self.model = LogisticRegressionModel(**cfg.get("model_kwargs", {}))
        elif model_name == "Polynomial":
            self.model = PolynomialRegressionModel(
                degree=cfg.get("degree", 2), **cfg.get("model_kwargs", {})
            )
        elif model_name == "DecisionTree":
            self.model = DecisionTreeModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "CART":
            self.model = CARTModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "ID3":
            self.model = ID3Model(
                task=cfg.get("task", "classification"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "C45":
            self.model = C45Model(
                task=cfg.get("task", "classification"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "C50":
            self.model = C50Model(
                task=cfg.get("task", "classification"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "CHAID":
            self.model = CHAIDModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "DecisionStump":
            self.model = DecisionStumpModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "ConditionalTree":
            self.model = ConditionalDecisionTreeModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "M5":
            self.model = M5Model(**cfg.get("model_kwargs", {}))
        elif model_name == "RandomForest":
            self.model = RandomForestModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "GradientBoosting" or model_name == "GBM":
            self.model = GradientBoostingModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "GBRT":
             self.model = GBRTModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "AdaBoost":
             self.model = AdaBoostModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "Bagging":
             self.model = BaggingModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "Stacking":
             self.model = StackingModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "Voting":
             self.model = VotingModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "WeightedAverage" or model_name == "Blending":
             self.model = WeightedAverageModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "XGBoost":
            self.model = XGBoostModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "LightGBM":
            self.model = LightGBMModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "kNN":
            self.model = kNNModel(
                task=cfg.get("task", "regression"),
                n_neighbors=cfg.get("n_neighbors", 5),
                **cfg.get("model_kwargs", {}),
            )
        elif model_name == "SVM":
            self.model = SVMModel(
                task=cfg.get("task", "regression"),
                kernel=cfg.get("kernel", "rbf"),
                **cfg.get("model_kwargs", {}),
            )
        elif model_name == "SVR":
            self.model = SVRModel(**cfg.get("model_kwargs", {}))
        elif model_name == "LinearSVM":
            self.model = LinearSVMModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "NuSVM":
            self.model = NuSVMModel(
                task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "OneClassSVM":
            self.model = OneClassSVMModel(**cfg.get("model_kwargs", {}))
        elif model_name == "LSSVM":
            self.model = LSSVMModel(**cfg.get("model_kwargs", {}))
        elif model_name == "TWSVM":
            self.model = TWSVMModel(**cfg.get("model_kwargs", {}))
        elif model_name == "NaiveBayes":
            self.model = NaiveBayesModel(
                type=cfg.get("type", "gaussian"), **cfg.get("model_kwargs", {})
            )
        elif model_name == "GaussianNB":
            self.model = GaussianNaiveBayesModel(**cfg.get("model_kwargs", {}))
        elif model_name == "MultinomialNB":
            self.model = MultinomialNaiveBayesModel(**cfg.get("model_kwargs", {}))
        elif model_name == "AODE":
            self.model = AODEModel(**cfg.get("model_kwargs", {}))
        elif model_name == "BayesianNetwork" or model_name == "BBN" or model_name == "BN":
            self.model = BayesianNetworkModel(**cfg.get("model_kwargs", {}))
        elif model_name == "OLSR":
            self.model = OLSRModel(**cfg.get("model_kwargs", {}))
        elif model_name == "Stepwise":
            self.model = StepwiseRegressionModel(**cfg.get("model_kwargs", {}))
        elif model_name == "MARS":
            self.model = MARSModel(**cfg.get("model_kwargs", {}))
        elif model_name == "LOESS":
            self.model = LOESSModel(**cfg.get("model_kwargs", {}))
        elif model_name == "LWL":
            self.model = LWLModel(
                task=cfg.get("task", "regression"),
                n_neighbors=cfg.get("n_neighbors", 5),
                **cfg.get("model_kwargs", {}),
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor or dictionary containing 'observation'.

        Returns:
            Model output.
        """
        if hasattr(x, "get"):
            x = x.get("observation")

        kwargs = {}
        if self.cfg.get("return_sequence", False):
            kwargs["return_sequence"] = True

        # Whitelist for models supporting return_sequence
        # All our new models I just refactored support it.
        sequence_supported = [
            "LSTM",
            "GRU",
            "Mamba",
            "xLSTM",
            "SNN",
            "ESN",
            "MLP",
            "ELM",
            "RBF",
            "AE",
            "DAE",
            "SAE",
            "StackedAE",
            "Hopfield",
            "RBM",
            "SOM",
            "Capsule",
            "Perceptron",
            "MarkovChain",
            "BM",
            "DBN",
            "DCN",
            "Deconv",
            "AutoDeconv",
            "DCIGN",
            "LSM",
            "ResNet",
            "DNC",
            "NTM",
            "Attention",
            "Flow",
            "NODE",
            "PINN",
            "LinearRegression",
            "Ridge",
            "Lasso",
            "LARS",
            "ElasticNet",
            "LogisticRegression",
            "Polynomial",
            "DecisionTree",
            "RandomForest",
            "GradientBoosting",
            "GBM",
            "GBRT",
            "AdaBoost",
            "Bagging",
            "Stacking",
            "Voting",
            "WeightedAverage",
            "Blending",
            "XGBoost",
            "LightGBM",
            "kNN",
            "SVM",
            "SVM",
            "SVR",
            "LinearSVM",
            "NuSVM",
            "OneClassSVM",
            "LSSVM",
            "TWSVM",
            "NaiveBayes",
            "GaussianNB",
            "MultinomialNB",
            "AODE",
            "BayesianNetwork",
            "BBN",
            "BN",
            "OLSR",
            "Stepwise",
            "MARS",
            "LOESS",
        ]

        if self.cfg.get("name") in sequence_supported:
            pass
        elif "return_sequence" in kwargs:
            del kwargs["return_sequence"]

        out = self.model(x, **kwargs)
        return out
