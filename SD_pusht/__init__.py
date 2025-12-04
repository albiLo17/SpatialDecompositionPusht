"""SD_pusht: Spatial Decomposition for PushT using Diffusion Policy."""

import os

# Define root directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(os.path.dirname(ROOT_DIR), "datasets")
RUNS_DIR = os.path.join(os.path.dirname(ROOT_DIR), "runs")

# Expose main APIs
from SD_pusht.models import ConditionalUnet1D
from SD_pusht.datasets import PushTStateDataset
from SD_pusht.utils import (
    get_data_stats,
    normalize_data,
    unnormalize_data,
    make_env,
    apply_legacy_state,
    tile_images,
    evaluate_model,
)

__all__ = [
    "ConditionalUnet1D",
    "PushTStateDataset",
    "get_data_stats",
    "normalize_data",
    "unnormalize_data",
    "make_env",
    "apply_legacy_state",
    "tile_images",
    "evaluate_model",
    "ROOT_DIR",
    "DATASETS_DIR",
    "RUNS_DIR",
]

# Backward compatibility: support old import paths
# These will be deprecated but kept for now
import sys
from pathlib import Path

# Add backward compatibility imports by creating stub modules
_old_imports = {
    "network": "SD_pusht.models.conditional_unet",
    "push_t_dataset": "SD_pusht.datasets.push_t_dataset",
}
