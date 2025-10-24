# PushT Segment Visualization Guide

This guide explains how to visualize segmented PushT trajectories with environment rendering and trajectory tracking.

## Overview

The unified visualization tool (`render_segments.py`) provides multiple rendering modes:
- **Single segment rendering** - Interactive or save frames
- **Animated GIFs** - Full length segments with trajectory overlays
- **Comparison grids** - Side-by-side visualization of multiple segments
- **Batch GIF creation** - Process multiple segments at once

All modes support:
- Contact vs no-contact segment differentiation with color coding
- Full state reconstruction from original zarr dataset
- Trajectory path visualization
- Block and agent tracking

## Quick Start

```bash
# Render a single segment interactively
python SD_pusht/render_segments.py single --segment-index 0

# Create an animated GIF with trajectory overlay
python SD_pusht/render_segments.py gif --segment-index 0 --output segment_0.gif

# Create comparison grid of multiple segments
python SD_pusht/render_segments.py comparison --segment-indices 0 1 2 3 --save-path comparison.png

# Batch create GIFs for contact segments only
python SD_pusht/render_segments.py batch-gif --contact-only --max-segments 5
```

## Rendering Modes

### 1. Single Segment Rendering

Render a specific segment with environment visualization.

```bash
# Interactive rendering (human mode)
python SD_pusht/render_segments.py single --segment-index 0 --speed 1.0

# Save frames as images
python SD_pusht/render_segments.py single \
  --segment-index 0 \
  --render-mode rgb_array \
  --save-images \
  --output-dir rendered_segments
```

**Options:**
- `--segment-index`: Segment to render (required)
- `--render-mode`: `human` (interactive) or `rgb_array` (save frames)
- `--speed`: Playback speed multiplier (default: 1.0)
- `--save-images`: Save rendered frames
- `--output-dir`: Directory for saved images

### 2. Animated GIF Creation

Create polished GIFs with trajectory overlays.

```bash
# Basic GIF
python SD_pusht/render_segments.py gif \
  --segment-index 0 \
  --output segment_0.gif

# High-quality GIF with block trajectory
python SD_pusht/render_segments.py gif \
  --segment-index 0 \
  --output segment_0.gif \
  --fps 15 \
  --trajectory-length 50 \
  --show-block-trajectory

# Reduced file size
python SD_pusht/render_segments.py gif \
  --segment-index 0 \
  --output segment_0.gif \
  --downsample 2
```

**Options:**
- `--segment-index`: Segment to render (required)
- `--output`: Output GIF path (required)
- `--fps`: Frames per second (default: 10)
- `--trajectory-length`: Number of past positions to show (default: 30)
- `--downsample`: Use every Nth frame (default: 1)
- `--show-block-trajectory`: Also show block path

### 3. Comparison Grid

Create side-by-side comparisons of multiple segments.

```bash
# Specific segments
python SD_pusht/render_segments.py comparison \
  --segment-indices 0 1 2 3 \
  --save-path comparison.png

# Contact segments only
python SD_pusht/render_segments.py comparison \
  --contact-only \
  --max-segments 6 \
  --save-path contact_comparison.png

# No-contact segments only
python SD_pusht/render_segments.py comparison \
  --no-contact-only \
  --max-segments 6 \
  --save-path no_contact_comparison.png

# Interactive display (no save)
python SD_pusht/render_segments.py comparison \
  --segment-indices 0 1 2 3
```

**Options:**
- `--segment-indices`: Specific segments to show
- `--max-segments`: Maximum segments if indices not specified (default: 6)
- `--save-path`: Save to file (if not provided, shows interactively)
- `--contact-only`: Only show contact segments
- `--no-contact-only`: Only show no-contact segments

### 4. Batch GIF Creation

Create GIFs for multiple segments at once.

```bash
# Contact segments only
python SD_pusht/render_segments.py batch-gif \
  --contact-only \
  --max-segments 10 \
  --output-dir contact_gifs

# No-contact segments only
python SD_pusht/render_segments.py batch-gif \
  --no-contact-only \
  --max-segments 10 \
  --output-dir no_contact_gifs

# Specific segments
python SD_pusht/render_segments.py batch-gif \
  --segment-indices 0 5 10 15 \
  --output-dir selected_gifs \
  --fps 12 \
  --show-block-trajectory
```

**Options:**
- `--segment-indices`: Specific segments to render
- `--max-segments`: Maximum segments if indices not specified (default: 4)
- `--output-dir`: Directory for output GIFs (default: `segment_gifs`)
- `--fps`: Frames per second (default: 10)
- `--trajectory-length`: Trajectory trail length (default: 30)
- `--downsample`: Frame sampling (default: 1)
- `--show-block-trajectory`: Include block trajectory
- `--contact-only`: Only contact segments
- `--no-contact-only`: Only no-contact segments

## Common Use Cases

### Quick Debug: View a Segment Interactively

```bash
python SD_pusht/render_segments.py single --segment-index 0
```

### Create Publication-Quality GIF

```bash
python SD_pusht/render_segments.py gif \
  --segment-index 0 \
  --output figures/segment_0_demo.gif \
  --fps 15 \
  --trajectory-length 50 \
  --show-block-trajectory
```

### Analyze Contact vs No-Contact Patterns

```bash
# Create comparison grid
python SD_pusht/render_segments.py comparison \
  --contact-only \
  --max-segments 4 \
  --save-path contact_analysis.png

python SD_pusht/render_segments.py comparison \
  --no-contact-only \
  --max-segments 4 \
  --save-path no_contact_analysis.png
```

### Create Dataset Overview

```bash
# Batch create GIFs for first 20 segments
python SD_pusht/render_segments.py batch-gif \
  --max-segments 20 \
  --fps 10 \
  --downsample 2 \
  --output-dir dataset_overview
```

### Memory-Efficient Large Batch

```bash
# Process many segments with reduced file sizes
python SD_pusht/render_segments.py batch-gif \
  --max-segments 50 \
  --fps 8 \
  --trajectory-length 20 \
  --downsample 3 \
  --output-dir large_batch
```

## Output Formats

### GIF Files (batch-gif mode)

Saved with descriptive names:
- `segment_000_no_contact.gif`
- `segment_001_contact.gif`
- `segment_002_contact.gif`

Each GIF includes:
- Environment rendering (512x512 pixels)
- Trajectory overlay with gradient
- Text overlay (segment type, frame number, episode)
- Smooth animation

### Image Grids (comparison mode)

PNG files with multiple segments side-by-side:
- Grid layout (up to 3 columns)
- Segment info as titles
- First frame of each segment

### Image Sequences (single mode with --save-images)

Individual PNG frames:
- `segment_000_frame_000.png`
- `segment_000_frame_001.png`
- etc.

## Visual Features

### Color Coding

- **Agent Trajectory**:
  - Red gradient: Contact segments (agent pushing object)
  - Blue gradient: No-contact segments (agent moving alone)
- **Block Trajectory** (when `--show-block-trajectory`):
  - Green gradient
- **Current Positions**:
  - Yellow circle: Agent's current position
  - Green circle: Block's current position (if block trajectory enabled)
- **Background**: Full PushT environment rendering

### Trajectory Gradient

The trajectory trail uses alpha blending:
- Older positions: Faint/transparent
- Recent positions: Bright/opaque
- Clearly shows movement direction and speed

## File Sizes & Performance

Typical GIF file sizes:
- 20 frames @ 10 fps ≈ 150-200 KB
- 50 frames @ 10 fps ≈ 250-350 KB
- 100 frames @ 10 fps ≈ 500-700 KB

**Reduce file size:**
- Increase `--downsample` (try 2 or 3)
- Reduce `--trajectory-length`
- Reduce `--fps`

**Increase quality:**
- Use `--downsample 1` (all frames)
- Increase `--fps` (15-20 for smooth playback)
- Increase `--trajectory-length` (40-50 for long trails)

## Best Practices

### For Detailed Analysis
```bash
python SD_pusht/render_segments.py gif \
  --segment-index 0 \
  --output analysis.gif \
  --fps 15 \
  --trajectory-length 50 \
  --show-block-trajectory
```
High fps (15-20), long trajectory (40-50), no downsampling

### For Quick Overview
```bash
python SD_pusht/render_segments.py batch-gif \
  --max-segments 20 \
  --fps 8 \
  --trajectory-length 20 \
  --downsample 2 \
  --output-dir overview
```
Low fps (5-10), short trajectory (15-20), downsample 2-3

### For Presentations
```bash
python SD_pusht/render_segments.py gif \
  --segment-index 0 \
  --output presentation.gif \
  --fps 12 \
  --trajectory-length 30 \
  --show-block-trajectory
```
Medium fps (10-12), medium trajectory (25-30), show block trajectory

### For File Size Optimization
```bash
python SD_pusht/render_segments.py batch-gif \
  --max-segments 100 \
  --fps 8 \
  --trajectory-length 15 \
  --downsample 3 \
  --output-dir optimized
```
Increase downsampling, reduce trajectory length, reduce fps

## Dataset Configuration

All modes use these default paths (can be overridden):
- `--segmented-dataset`: `datasets/pusht_segmented_clean_fixed.npz`
- `--metadata`: `datasets/pusht_segmented_clean_metadata.npz`
- `--original-dataset`: `datasets/pusht_cchi_v7_replay.zarr.zip`

Override example:
```bash
python SD_pusht/render_segments.py single \
  --segment-index 0 \
  --segmented-dataset path/to/segments.npz \
  --metadata path/to/metadata.npz \
  --original-dataset path/to/original.zarr.zip
```

## Integration Details

The visualization tool:
- Loads segment metadata automatically from npz files
- Reconstructs full states (5D: agent_x, agent_y, block_x, block_y, block_angle) from original zarr dataset
- Maintains temporal accuracy (uses actual timesteps from episodes)
- Preserves all segment information (episode index, contact flag, segment length, timestep ranges)
- Uses PushT environment's legacy rendering mode for accurate visualization

## Troubleshooting

**Error: "segment_index X is out of range"**
```bash
# Check how many segments you have
python -c "import numpy as np; data = np.load('datasets/pusht_segmented_clean_metadata.npz', allow_pickle=True); print(f'Total segments: {len(data[\"segment_metadata\"])}')"
```

**GIFs too large?**
- Increase `--downsample` (try 2 or 3)
- Reduce `--trajectory-length` (try 15-20)
- Reduce `--fps` (try 8)

**Animation too fast/slow?**
- Adjust `--fps` (range: 5-20)
- For single mode human rendering, adjust `--speed` multiplier

**Trajectory too short/long?**
- Adjust `--trajectory-length` (range: 15-50)

**Want to see object movement?**
- Add `--show-block-trajectory` flag

**Need help with commands?**
```bash
# General help
python SD_pusht/render_segments.py --help

# Mode-specific help
python SD_pusht/render_segments.py single --help
python SD_pusht/render_segments.py gif --help
python SD_pusht/render_segments.py comparison --help
python SD_pusht/render_segments.py batch-gif --help
```

## Advanced Usage

### Process All Segments

```bash
# First, get the total count
TOTAL=$(python -c "import numpy as np; data = np.load('datasets/pusht_segmented_clean_metadata.npz', allow_pickle=True); print(len(data['segment_metadata']))")

# Create GIFs for all segments
python SD_pusht/render_segments.py batch-gif \
  --max-segments $TOTAL \
  --fps 10 \
  --output-dir all_segments
```

### Custom Trajectory Colors

The visualization uses these colors (can be modified in `render_segments.py`):
- Contact segments: Red gradient (`(255, alpha, alpha)`)
- No-contact segments: Blue gradient (`(alpha, alpha, 255)`)
- Block trajectory: Green gradient (`(alpha, 255, alpha)`)
- Current agent position: Yellow circle
- Current block position: Green circle

To customize, edit the color values in the `create_segment_gif()` function.

## Script Architecture

The unified `render_segments.py` contains:
1. **Utility functions**: State loading, coordinate conversion, environment control
2. **Rendering modes**: Single, GIF, comparison, batch-gif
3. **Command-line interface**: Argparse with subcommands for each mode
4. **Shared codebase**: All modes use the same core rendering functions

This consolidation eliminates code duplication and provides a consistent interface.

## Related Files

- `SD_pusht/render_segments.py`: **Main unified rendering script** (use this!)
- `SD_pusht/notebooks/visualize_pusht_dataset.ipynb`: Interactive Jupyter notebook for visualization

