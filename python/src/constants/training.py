
from __future__ import annotations

import os

# Hyperparameters Defaults
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "0.01"))
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_EPOCHS = 10
DEFAULT_SEQ_LEN = 30
DEFAULT_PRED_LEN = 1

# Labels
TARGET_COLUMN = "price"
TIMESTAMP_COLUMN = "timestamp"

# Training Modes
MODE_TRAIN = "train"
MODE_VAL = "val"
MODE_TEST = "test"
