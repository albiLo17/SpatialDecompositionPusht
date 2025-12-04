# Cleanup Summary

## Files Removed

The following old script files were removed as they are now duplicated in the `scripts/` directory:

- ✅ `train_diffusion.py` → Now redirects to `scripts/train.py`
- ✅ `eval_diffusion.py` → Import stub only (redirects imports, not execution)
- ✅ `replay_demos.py` → Use `scripts/replay_demos.py`
- ✅ `collect_demos.py` → Use `scripts/collect_demos.py`
- ✅ `convert_pusht_to_toydataset.py` → Use `scripts/convert_pusht_to_toydataset.py`
- ✅ `convert_pusht_segmented.py` → Use `scripts/convert_pusht_segmented.py`
- ✅ `render_segments.py` → Use `scripts/render_segments.py`

## Files Kept (Backward Compatibility)

These are lightweight stubs that redirect imports:

- ✅ `network.py` → Redirects imports to `SD_pusht.models`
- ✅ `push_t_dataset.py` → Redirects imports to `SD_pusht.datasets`
- ✅ `eval_diffusion.py` → Redirects imports to `SD_pusht.utils.evaluation`
- ✅ `train_diffusion.py` → Redirects execution to `scripts/train.py`

## Migration

### For Direct Script Execution

**Old way:**
```bash
python SD_pusht/train_diffusion.py --epochs 100
python SD_pusht/eval_diffusion.py --ckpt-path ...
```

**New way (recommended):**
```bash
python SD_pusht/scripts/train.py --epochs 100
python SD_pusht/scripts/eval.py --ckpt-path ...
```

Or using module syntax:
```bash
python -m SD_pusht.scripts.train --epochs 100
python -m SD_pusht.scripts.eval --ckpt-path ...
```

### For Imports in Python Code

**Old way (still works with warnings):**
```python
from network import ConditionalUnet1D
from push_t_dataset import PushTStateDataset
```

**New way (recommended):**
```python
from SD_pusht.models import ConditionalUnet1D
from SD_pusht.datasets import PushTStateDataset
```

## Current Structure

```
SD_pusht/
├── scripts/              # All executable scripts (USE THESE)
│   ├── train.py
│   ├── eval.py
│   ├── replay_demos.py
│   └── ...
├── network.py            # Import stub (backward compat)
├── push_t_dataset.py     # Import stub (backward compat)
├── eval_diffusion.py     # Import stub (backward compat)
└── train_diffusion.py    # Execution stub (backward compat)
```

