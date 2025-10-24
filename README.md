# Spatial Decomposition for PushT

## Description
Spatial Decomposition from SuperN1ck Evaluated on the PushT environment.

## Installation
These steps create a conda environment from the repo, install the `uv` wrapper (if available) and use it to run `pip install -e .` for faster installs. If `uv` is not available on your environment, the fallback `pip install -e .` is shown.

1. From the repository root, create a minimal environment manually:
```bash
conda create -n SD_pusht python=3.12 -y
conda activate SD_pusht
```

2. Install `uv` (optional speed-up wrapper) and verify it is available:
```bash
pip install uv 
```
Note: if `uv` is not published / available on your system, skip this step and use `pip` below.

3. Initialize submodules (if your repo uses them) and install them editable:
```bash
git submodule update --init --recursive

# install submodules (example — run for each submodule path)
uv pip install -e gym_pusht 
uv pip install -e spatialdecomposition
uv pip install -e casino
uv pip install -e robosuite
uv pip install -e robomimic
uv pip install -e libero
uv pip install -e diffusion_policy
```

4. Install the main package in editable (development) mode:
```bash
# use uv if available (faster), otherwise use pip
uv pip install -e . 
```

5. PyTorch note (optional)
For best results with `torch`/`torchvision`/`torchaudio` (CUDA vs CPU wheels), install the correct wheels first following https://pytorch.org/ and then run steps above.

6. Additional module for visualization and rendering in the cluster:
```bash
conda install -c conda-forge ffmpeg=6.* "libstdcxx-ng>=12" -y
```

## Usage
TBD, but ideally the steps are: download dataset, train DP to see if it works, install Nick's repo and integrate it to segment the demos, adjust and train the model.

1. Check if the demo dataset exists, otherwise download it:
```bash
python SD_pusht/push_t_dataset.py 
```
2. Visualize one demonstration:
```bash
python SD_pusht/replay_demos.py -e 4
```
3. Convert the dataset into ToyDataset and Segment it by contacts with the object:
```bash
python SD_pusht/convert_pusht_to_toydataset.py \
    --input datasets/pusht_cchi_v7_replay.zarr.zip \
    --output datasets/pusht_toy_dataset_segmented.npz \
    --traj-length 64 --max-episodes 1000 

```