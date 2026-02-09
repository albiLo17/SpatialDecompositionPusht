#!/usr/bin/env python3
"""
Plot evaluation results for diffusion, flow matching, and local flow 2D models.

This script reads evaluation results from model folders and creates two plots:
1. Success Rate vs Number of Demos (with separate curves for diffusion, flow matching, and local flow 2D)
2. Mean Max Single-Step Reward vs Number of Demos (with separate curves for diffusion, flow matching, and local flow 2D)

The plots are grouped by dataset dimensions (obs_dim, action_dim) if they differ across models.
"""

import os
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
import json


def parse_model_name(folder_name):
    """
    Parse model folder name to extract model type and number of demos.
    
    Examples:
    - sd-pusht-diffusion-demos-200-seed1234 -> ('diffusion', 200)
    - sd-pusht-fm-demos-100-seed1234 -> ('flow_matching', 100)
    - sd-pusht-local-flow-2d_V3-demos-100-seed1234-no-share-noise-pred-ref-concat-median -> ('local_flow_2d_median_concat', 100)
    - sd-pusht-local-flow-2d_V3-demos-100-seed1234-no-share-noise-pred-ref-film-knn -> ('local_flow_2d_knn_film', 100)
    """
    # Match pattern: sd-pusht-{type}-demos-{num}-seed{seed}
    match = re.match(r'sd-pusht-(diffusion|fm)-demos-(\d+)-seed\d+', folder_name)
    if match:
        model_type = 'diffusion' if match.group(1) == 'diffusion' else 'flow_matching'
        num_demos = int(match.group(2))
        return model_type, num_demos
    
    # Match pattern for local flow 2D with aggregation and FiLM: 
    # Format: sd-pusht-local-flow-2d_V3-demos-{num}-seed{seed}-{noise}-{ref}-{film}-{agg}
    # Examples: ...-gt-ref-concat-median, ...-gt-ref-film-knn
    match = re.match(r'sd-pusht-local-flow-2d_V\d+-demos-(\d+)-seed\d+-(?:no-)?share-noise-(?:gt-|pred-)ref-(concat|film)-(median|knn)', folder_name)
    if match:
        num_demos = int(match.group(1))
        film_str = match.group(2)  # 'concat' or 'film'
        agg_str = match.group(3)   # 'median' or 'knn'
        model_type = f'local_flow_2d_{agg_str}_{film_str}'
        return model_type, num_demos
    
    # Match pattern for local flow 2D with only FiLM (no aggregation suffix, defaults to median)
    # Format: sd-pusht-local-flow-2d_V3-demos-{num}-seed{seed}-{noise}-{ref}-{film}
    # Examples: ...-gt-ref-concat, ...-gt-ref-film
    match = re.match(r'sd-pusht-local-flow-2d_V\d+-demos-(\d+)-seed\d+-(?:no-)?share-noise-(?:gt-|pred-)ref-(concat|film)$', folder_name)
    if match:
        num_demos = int(match.group(1))
        film_str = match.group(2)  # 'concat' or 'film'
        agg_str = 'median'  # Default to median if not specified
        model_type = f'local_flow_2d_{agg_str}_{film_str}'
        return model_type, num_demos
    
    # Also match if film-agg are in different order (shouldn't happen, but just in case)
    match = re.match(r'sd-pusht-local-flow-2d_V\d+-demos-(\d+)-seed\d+-(?:no-)?share-noise-(?:gt-|pred-)ref-(median|knn)-(concat|film)', folder_name)
    if match:
        num_demos = int(match.group(1))
        agg_str = match.group(2)   # 'median' or 'knn'
        film_str = match.group(3)  # 'concat' or 'film'
        model_type = f'local_flow_2d_{agg_str}_{film_str}'
        return model_type, num_demos
    
    # Fallback: Match pattern for older local flow 2D without aggregation/FiLM suffix
    # Format: sd-pusht-local-flow-2d_V3-demos-{num}-seed{seed}-{noise}-{ref} or just ...-seed{seed}
    match = re.match(r'sd-pusht-local-flow-2d_V\d+-demos-(\d+)-seed\d+', folder_name)
    if match:
        model_type = 'local_flow_2d'
        num_demos = int(match.group(1))
        return model_type, num_demos
    
    return None, None


def load_eval_results(base_dir="log/dp"):
    """
    Load all evaluation results from model folders.
    
    Returns:
        dict: {dimension_key: {model_type: [(num_demos, success_rate, mean_max_reward), ...]}}
        where dimension_key is a string like "obs_dim=5,action_dim=2"
    """
    results = defaultdict(lambda: defaultdict(list))
    
    # Convert to absolute path if relative (relative to project root)
    if not os.path.isabs(base_dir):
        # Get project root (two levels up from this script)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        base_path = project_root / base_dir
    else:
        base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Warning: Base directory {base_path} does not exist!")
        return results
    
    # Find all model folders
    for model_folder in base_path.iterdir():
        if not model_folder.is_dir():
            continue
        
        # Check if eval_results directory exists
        eval_results_dir = model_folder / "eval_results"
        npz_path = eval_results_dir / "eval_results.npz"
        
        if not npz_path.exists():
            continue
        
        # Load npz file
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # Always parse model type from folder name first (has the most detailed info)
            # The npz file might only have generic 'local_flow_2d', but folder name has variants
            model_type, num_demos = parse_model_name(model_folder.name)
            
            if model_type is None or num_demos is None:
                # Fallback: try to get from npz file if folder parsing fails
                if 'model_type' in data:
                    model_type_from_npz = str(data['model_type'].item())
                    if model_type_from_npz == 'local_flow_2d':
                        model_type = 'local_flow_2d'
                    elif model_type_from_npz == 'flow_matching':
                        model_type = 'flow_matching'
                    elif model_type_from_npz == 'diffusion':
                        model_type = 'diffusion'
                
                # Still need num_demos
                if num_demos is None:
                    # Try to get from config
                    config_path = model_folder / "config.json"
                    if config_path.exists():
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            num_demos = config.get('max_demos', None)
                        except Exception:
                            pass
                    
                    if num_demos is None:
                        print(f"Warning: Could not determine num_demos for: {model_folder.name}")
                        continue
                
                if model_type is None:
                    print(f"Warning: Could not parse model name: {model_folder.name}")
                    continue
            
            # Extract metrics
            success_rate = float(data['success_rate'])
            
            # Try to get mean_max_single_reward, use 0.0 if not available
            if 'mean_max_single_reward' in data:
                mean_max_single_reward = float(data['mean_max_single_reward'])
            else:
                print(f"Warning: mean_max_single_reward not found in {npz_path}, skipping...")
                continue
            
            # Get dataset dimensions from config.json if available
            config_path = model_folder / "config.json"
            obs_dim = None
            action_dim = None
            
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    obs_dim = config.get('obs_dim', None)
                    action_dim = config.get('action_dim', None)
                except Exception as e:
                    print(f"Warning: Could not load config from {config_path}: {e}")
            
            # If not in config, try to get from npz
            if obs_dim is None:
                obs_dim = int(data.get('obs_dim', 5))
            if action_dim is None:
                action_dim = int(data.get('action_dim', 2))
            
            # Create dimension key
            dim_key = f"obs_dim={obs_dim},action_dim={action_dim}"
            
            # Store results
            results[dim_key][model_type].append((num_demos, success_rate, mean_max_single_reward))
            
        except Exception as e:
            print(f"Warning: Could not load results from {npz_path}: {e}")
            continue
    
    # Sort by num_demos for each model type
    for dim_key in results:
        for model_type in results[dim_key]:
            results[dim_key][model_type].sort(key=lambda x: x[0])
    
    return results


def create_plots(results, output_dir="plots"):
    """
    Create plots for evaluation results.
    
    Args:
        results: dict from load_eval_results()
        output_dir: directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for plotting. Please install it with: pip install matplotlib")
        return
    
    # Convert to absolute path if relative (relative to project root)
    if not os.path.isabs(output_dir):
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        output_dir = str(project_root / output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Colors for different model types - using distinct colors from matplotlib's tab10 palette
    colors = {
        'diffusion': '#1f77b4',  # blue
        'flow_matching': '#ff7f0e',  # orange
        'local_flow_2d': '#2ca02c',  # green (legacy)
        'local_flow_2d_median_concat': '#2ca02c',  # green (baseline - median aggregation, concatenation)
        'local_flow_2d_knn_concat': '#d62728',  # red (KNN aggregation, concatenation)
        'local_flow_2d_median_film': '#9467bd',  # purple (median aggregation, FiLM)
        'local_flow_2d_knn_film': '#8c564b'  # brown (KNN aggregation, FiLM)
    }
    
    # Labels for model types - more descriptive
    labels = {
        'diffusion': 'Diffusion Policy',
        'flow_matching': 'Flow Matching Policy',
        'local_flow_2d': 'Local Flow 2D (Legacy)',
        'local_flow_2d_median_concat': 'Local Flow 2D: Median Aggregation, Concatenation',
        'local_flow_2d_knn_concat': 'Local Flow 2D: KNN Aggregation, Concatenation',
        'local_flow_2d_median_film': 'Local Flow 2D: Median Aggregation, FiLM Conditioning',
        'local_flow_2d_knn_film': 'Local Flow 2D: KNN Aggregation, FiLM Conditioning'
    }
    
    # Create plots for each dimension group
    for dim_key, dim_results in results.items():
        # Extract data for plotting
        diffusion_data = dim_results.get('diffusion', [])
        flow_matching_data = dim_results.get('flow_matching', [])
        local_flow_2d_data = dim_results.get('local_flow_2d', [])
        local_flow_2d_median_concat_data = dim_results.get('local_flow_2d_median_concat', [])
        local_flow_2d_knn_concat_data = dim_results.get('local_flow_2d_knn_concat', [])
        local_flow_2d_median_film_data = dim_results.get('local_flow_2d_median_film', [])
        local_flow_2d_knn_film_data = dim_results.get('local_flow_2d_knn_film', [])
        
        # Check if we have any data to plot
        all_data = [diffusion_data, flow_matching_data, local_flow_2d_data,
                   local_flow_2d_median_concat_data, local_flow_2d_knn_concat_data,
                   local_flow_2d_median_film_data, local_flow_2d_knn_film_data]
        if not any(all_data):
            continue
        
        # Prepare data for plotting - ensure data is sorted by num_demos
        plot_data = {}
        for model_type, data_list in dim_results.items():
            if data_list:
                # Sort by num_demos to ensure proper line plotting
                sorted_data = sorted(data_list, key=lambda x: x[0])
                num_demos = [x[0] for x in sorted_data]
                success_rates = [x[1] for x in sorted_data]
                mean_max_rewards = [x[2] for x in sorted_data]
                # Only plot if we have at least one data point
                if len(num_demos) > 0:
                    plot_data[model_type] = {
                        'num_demos': num_demos,
                        'success_rates': success_rates,
                        'mean_max_rewards': mean_max_rewards
                    }
        
        # Create figure with two subplots - make it wider to accommodate legend
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Success Rate vs Number of Demos
        # Plot in order: diffusion, flow_matching, then local flow 2D variants
        plot_order = ['diffusion', 'flow_matching', 'local_flow_2d',
                     'local_flow_2d_median_concat', 'local_flow_2d_knn_concat',
                     'local_flow_2d_median_film', 'local_flow_2d_knn_film']
        
                
        # Plot models in predefined order first
        for model_type in plot_order:
            if model_type in plot_data:
                data = plot_data[model_type]
                ax1.plot(
                    data['num_demos'],
                    data['success_rates'],
                    'o-',
                    color=colors.get(model_type, '#000000'),
                    label=labels.get(model_type, model_type),
                    linewidth=2.5,
                    markersize=8
                )
        # Plot any remaining models not in predefined order
        for model_type, data in plot_data.items():
            if model_type not in plot_order:
                ax1.plot(
                    data['num_demos'],
                    data['success_rates'],
                    'o-',
                    color=colors.get(model_type, '#000000'),
                    label=labels.get(model_type, model_type),
                    linewidth=2.5,
                    markersize=8
                )
        
        ax1.set_xlabel('Number of Demos', fontsize=12)
        ax1.set_ylabel('Success Rate', fontsize=12)
        ax1.set_title(f'Success Rate vs Number of Demos\n({dim_key})', fontsize=13)
        ax1.legend(loc='best', fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Plot 2: Mean Max Single-Step Reward vs Number of Demos
        plot_order = ['diffusion', 'flow_matching', 'local_flow_2d',
                     'local_flow_2d_median_concat', 'local_flow_2d_knn_concat',
                     'local_flow_2d_median_film', 'local_flow_2d_knn_film']
        # Plot models in predefined order first
        for model_type in plot_order:
            if model_type in plot_data:
                data = plot_data[model_type]
                ax2.plot(
                    data['num_demos'],
                    data['mean_max_rewards'],
                    'o-',
                    color=colors.get(model_type, '#000000'),
                    label=labels.get(model_type, model_type),
                    linewidth=2.5,
                    markersize=8
                )
        # Plot any remaining models not in predefined order
        for model_type, data in plot_data.items():
            if model_type not in plot_order:
                ax2.plot(
                    data['num_demos'],
                    data['mean_max_rewards'],
                    'o-',
                    color=colors.get(model_type, '#000000'),
                    label=labels.get(model_type, model_type),
                    linewidth=2.5,
                    markersize=8
                )
        
        ax2.set_xlabel('Number of Demos', fontsize=12)
        ax2.set_ylabel('Mean Max Single-Step Reward', fontsize=12)
        ax2.set_title(f'Mean Max Single-Step Reward vs Number of Demos\n({dim_key})', fontsize=13)
        ax2.legend(loc='best', fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        # Sanitize dimension key for filename
        filename_dim_key = dim_key.replace(',', '_').replace('=', '')
        output_path = os.path.join(output_dir, f'eval_results_{filename_dim_key}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plots to: {output_path}")
        
        # Also save individual plots
        # Plot 1 only
        fig1, ax1_only = plt.subplots(1, 1, figsize=(10, 6))
        plot_order = ['diffusion', 'flow_matching', 'local_flow_2d',
                     'local_flow_2d_median_concat', 'local_flow_2d_knn_concat',
                     'local_flow_2d_median_film', 'local_flow_2d_knn_film']
        # Plot models in predefined order first
        for model_type in plot_order:
            if model_type in plot_data:
                data = plot_data[model_type]
                ax1_only.plot(
                    data['num_demos'],
                    data['success_rates'],
                    'o-',
                    color=colors.get(model_type, '#000000'),
                    label=labels.get(model_type, model_type),
                    linewidth=2.5,
                    markersize=8
                )
        # Plot any remaining models not in predefined order
        for model_type, data in plot_data.items():
            if model_type not in plot_order:
                ax1_only.plot(
                    data['num_demos'],
                    data['success_rates'],
                    'o-',
                    color=colors.get(model_type, '#000000'),
                    label=labels.get(model_type, model_type),
                    linewidth=2.5,
                    markersize=8
                )
        ax1_only.set_xlabel('Number of Demos', fontsize=12)
        ax1_only.set_ylabel('Success Rate', fontsize=12)
        ax1_only.set_title(f'Success Rate vs Number of Demos\n({dim_key})', fontsize=13)
        ax1_only.legend(loc='best', fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
        ax1_only.grid(True, alpha=0.3)
        ax1_only.set_ylim([0, 1.05])
        plt.tight_layout()
        output_path1 = os.path.join(output_dir, f'success_rate_{filename_dim_key}.png')
        plt.savefig(output_path1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Plot 2 only
        fig2, ax2_only = plt.subplots(1, 1, figsize=(10, 6))
        plot_order = ['diffusion', 'flow_matching', 'local_flow_2d',
                     'local_flow_2d_median_concat', 'local_flow_2d_knn_concat',
                     'local_flow_2d_median_film', 'local_flow_2d_knn_film']
        # Plot models in predefined order first
        for model_type in plot_order:
            if model_type in plot_data:
                data = plot_data[model_type]
                ax2_only.plot(
                    data['num_demos'],
                    data['mean_max_rewards'],
                    'o-',
                    color=colors.get(model_type, '#000000'),
                    label=labels.get(model_type, model_type),
                    linewidth=2.5,
                    markersize=8
                )
        # Plot any remaining models not in predefined order
        for model_type, data in plot_data.items():
            if model_type not in plot_order:
                ax2_only.plot(
                    data['num_demos'],
                    data['mean_max_rewards'],
                    'o-',
                    color=colors.get(model_type, '#000000'),
                    label=labels.get(model_type, model_type),
                    linewidth=2.5,
                    markersize=8
                )
        ax2_only.set_xlabel('Number of Demos', fontsize=12)
        ax2_only.set_ylabel('Mean Max Single-Step Reward', fontsize=12)
        ax2_only.set_title(f'Mean Max Single-Step Reward vs Number of Demos\n({dim_key})', fontsize=13)
        ax2_only.legend(loc='best', fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
        ax2_only.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path2 = os.path.join(output_dir, f'mean_max_reward_{filename_dim_key}.png')
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        plt.close(fig)


def main():
    """Main function to load results and create plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot evaluation results for diffusion, flow matching, and local flow 2D models")
    parser.add_argument("--base-dir", type=str, default="log/dp",
                       help="Base directory containing model folders (default: log/dp, relative to project root)")
    parser.add_argument("--output-dir", type=str, default="plots",
                       help="Output directory for plots (default: plots, relative to project root)")
    
    args = parser.parse_args()
    
    print(f"Loading evaluation results from: {args.base_dir}")
    results = load_eval_results(args.base_dir)
    
    if not results:
        print("No evaluation results found!")
        return
    
    print(f"\nFound results for {len(results)} dimension group(s):")
    for dim_key, dim_results in results.items():
        print(f"  {dim_key}:")
        for model_type, data_list in dim_results.items():
            print(f"    {model_type}: {len(data_list)} models")
    
    print(f"\nCreating plots...")
    create_plots(results, args.output_dir)
    print(f"\nDone! Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

