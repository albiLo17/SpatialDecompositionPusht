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
    tile_images,
    evaluate_model,
)

# Optional imports (require gym_pusht)
try:
    from SD_pusht.utils.environment import make_env, apply_legacy_state
except ImportError:
    make_env = None
    apply_legacy_state = None

__all__ = [
    "ConditionalUnet1D",
    "PushTStateDataset",
    "get_data_stats",
    "normalize_data",
    "unnormalize_data",
    "tile_images",
    "evaluate_model",
    "ROOT_DIR",
    "DATASETS_DIR",
    "RUNS_DIR",
]

# Add optional exports if available
if make_env is not None:
    __all__.append("make_env")
if apply_legacy_state is not None:
    __all__.append("apply_legacy_state")

# Backward compatibility: support old import paths
# These will be deprecated but kept for now
import sys
from pathlib import Path

# Add backward compatibility imports by creating stub modules
_old_imports = {
    "network": "SD_pusht.models.conditional_unet",
    "push_t_dataset": "SD_pusht.datasets.push_t_dataset",
}
