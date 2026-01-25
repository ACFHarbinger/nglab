
from __future__ import annotations

import os

# Data Directories
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CACHE_DIR = ".cache"

# Model Directories
MODELS_DIR = "models"
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
LOGS_DIR = "logs"

# Configs
CONFIG_DIR = "python/src/configs"

# Assets
# Assuming PROJECT_ROOT is available or we reconstruct it.
# To avoid circular imports, we can re-derive ROOT_DIR here or import carefully.
# Given this is 'paths', it should probably define the ROOT.
# definitions.py used getcwd(), let's use a robust relative path here.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ICON_FILE = os.path.join(ROOT_DIR, "assets", "images", "logo-nglab.png")
