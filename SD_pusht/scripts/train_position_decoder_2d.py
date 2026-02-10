#!/usr/bin/env python3
"""Training script for Position2DFlowDecoder (pose estimator) on PushT.

This script trains only the position decoder component, which predicts
reference positions from observations.

python SD_pusht/scripts/train_position_decoder_2d.py \
    --dataset-path datasets/pusht_cchi_v7_replay.zarr.zip \
    --max-demos 200 \
    --batch-size 256 \
    --epochs 100 \
    --wandb
    
# only mlp, no flow
python SD_pusht/scripts/train_position_decoder_2d.py \
    --backbone mlp_direct \
    --mlp-hidden-dims 256 512 256
    
# flow with mlp backbone
python SD_pusht/scripts/train_position_decoder_2d.py \
    --backbone mlp \
    --mlp-hidden-dims 256 512 256
"""

import argparse
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import gdown

from SD_pusht.models import Position2DFlowDecoder
from SD_pusht.datasets import PushTSegmentedDatasetSimple
from SD_pusht.datasets.softgym_segmented_dataset import SoftGymSegmentedDatasetSimple
from SD_pusht.utils.evaluation import visualize_position_predictions
from SD_pusht.utils.normalization import unnormalize_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train Position2DFlowDecoder on PushT")
    parser.add_argument("--dataset-path", type=str, 
                       default="datasets/pusht_cchi_v7_replay.zarr.zip",
                       help="Path to zarr dataset")
    parser.add_argument("--max-demos", type=int, default=None,
                       help="Maximum number of demonstration episodes to use for training")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--wandb", action="store_true", 
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="sd-pusht-position-decoder",
                       help="WandB project name")
    parser.add_argument("--eval-every", type=int, default=10,
                       help="Evaluate every N epochs (0 = no evaluation during training)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--obs-horizon", type=int, default=2,
                       help="Observation horizon")
    parser.add_argument("--obs-dim", type=int, default=5,
                       help="Observation dimension")
    parser.add_argument("--pred-horizon", type=int, default=16,
                       help="Prediction horizon")
    parser.add_argument("--action-horizon", type=int, default=8,
                       help="Action horizon")
    parser.add_argument("--sigma", type=float, default=0.0,
                       help="Noise scale for flow matching paths")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-6,
                       help="Weight decay")
    parser.add_argument("--ema-power", type=float, default=0.75,
                       help="EMA power parameter")
    parser.add_argument("--output-dir", type=str, default="log/dp",
                       help="Output directory for checkpoints and logs")
    # Position decoder specific arguments
    parser.add_argument("--backbone", type=str, default="unet", choices=["unet", "mlp", "mlp_direct"],
                       help="Backbone architecture: 'unet' (U-Net with flow), 'mlp' (MLP with flow), or 'mlp_direct' (MLP without flow)")
    parser.add_argument("--use-flow-matching", action="store_true", default=None,
                       help="Use flow matching (default: True for 'unet'/'mlp', False for 'mlp_direct')")
    parser.add_argument("--position-decoder-down-dims", type=int, nargs="+", default=[256],
                       help="Channel dimensions for position decoder U-Net (only used if backbone='unet')")
    parser.add_argument("--position-decoder-n-groups", type=int, default=4,
                       help="Number of groups for position decoder GroupNorm (only used if backbone='unet')")
    parser.add_argument("--position-decoder-fm-timesteps", type=int, default=20,
                       help="Number of timesteps for position decoder inference")
    parser.add_argument("--mlp-hidden-dims", type=int, nargs="+", default=[256, 512, 256],
                       help="Hidden layer dimensions for MLP backbone (only used if backbone='mlp')")
    parser.add_argument("--mlp-activation", type=str, default="mish", choices=["mish", "relu", "gelu", "tanh"],
                       help="Activation function for MLP backbone (only used if backbone='mlp')")
    # Dataset segmentation arguments
    parser.add_argument("--contact-threshold", type=float, default=0.1,
                       help="Threshold for detecting block movement (contact)")
    parser.add_argument("--min-segment-length", type=int, default=5,
                       help="Minimum length for a valid segment")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Fraction of episodes to use for validation")
    parser.add_argument("--val-min-episodes", type=int, default=10,
                       help="Minimum number of episodes for validation set")
    parser.add_argument("--eval-sample-seed", type=int, default=None,
                       help="Random seed for selecting evaluation sample (None = use epoch number)")
    # Observation noise augmentation (for regularization)
    parser.add_argument("--obs-noise-scale", type=float, default=0.01,
                       help="Scale of noise to add to observations during training (0.0 = no noise)")
    parser.add_argument("--obs-noise-type", type=str, default="gaussian", choices=["gaussian", "uniform"],
                       help="Type of noise to add: 'gaussian' or 'uniform'")
    # Multi-gripper support
    parser.add_argument("--num-pickers", type=int, default=2,
                       help="Number of grippers/pickers (default: 2)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine flow matching usage
    if args.use_flow_matching is None:
        # Auto-detect based on backbone
        use_flow_matching = args.backbone != "mlp_direct"
    else:
        use_flow_matching = args.use_flow_matching
    
    # Create experiment name
    flow_suffix = "flow" if use_flow_matching else "direct"
    noise_suffix = f"-noise{args.obs_noise_scale}" if args.obs_noise_scale > 0.0 else ""
    exp_name = f"sd-pusht-position-decoder-{args.backbone}-{flow_suffix}-demos-{args.max_demos}-seed{args.seed}{noise_suffix}"
    print(f"Experiment: {exp_name}")
    print(f"Using model: Position2DFlowDecoder with {args.backbone.upper()} backbone")
    print(f"Flow matching: {use_flow_matching}")
    if args.obs_noise_scale > 0.0:
        print(f"Observation noise: {args.obs_noise_type} with scale {args.obs_noise_scale}")
    
    # Create experiment directory and save config
    exp_dir = f"{args.output_dir}/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.json")
    
    # Convert args to dict and save as JSON
    config_dict = vars(args).copy()
    for key, value in config_dict.items():
        if not isinstance(value, (str, int, float, bool, type(None), list)):
            config_dict[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to: {config_path}")

    # optional wandb
    wandb_available = False
    if args.wandb:
        print("Using Weights & Biases logging")
        import wandb
        wandb_available = True
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.run.name = exp_name

    # download demonstration data from Google Drive if missing
    dataset_path = args.dataset_path
    if not os.path.isfile(dataset_path):
        print(f"Downloading dataset to {dataset_path}...")
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)

    # Load dataset info to get total number of episodes and metadata
    import zarr
    dataset_root = zarr.open(dataset_path, 'r')
    all_episode_ends = dataset_root['meta']['episode_ends'][:]
    total_episodes = len(all_episode_ends)
    meta = dataset_root['meta']
    
    # Get num_pickers from dataset metadata
    dataset_num_pickers = int(meta.attrs.get('num_picker', args.num_pickers))
    if dataset_num_pickers != args.num_pickers:
        print(f"Warning: Dataset has {dataset_num_pickers} pickers, but --num-pickers={args.num_pickers}. Using dataset value: {dataset_num_pickers}")
        args.num_pickers = dataset_num_pickers
    
    # Create training dataset (use SoftGymSegmentedDatasetSimple for SoftGym datasets)
    dataset = SoftGymSegmentedDatasetSimple(
        dataset_path=dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        max_demos=args.max_demos,
        use_gripper_segmentation=True,  # Use gripper-based segmentation
        use_contact_segmentation=False,
        min_segment_length=args.min_segment_length,
    )
    
    # Get demo indices used by training dataset
    train_demo_indices = getattr(dataset, 'demo_indices', None)
    if train_demo_indices is None:
        train_demo_indices = list(range(args.max_demos if args.max_demos is not None else total_episodes))
    
    print(f"Training dataset uses {len(train_demo_indices)} episodes")
    
    # Calculate validation demo indices
    val_dataset = None
    if args.val_split > 0:
        all_demo_indices = set(range(total_episodes))
        train_demo_indices_set = set(train_demo_indices)
        remaining_demo_indices = sorted(list(all_demo_indices - train_demo_indices_set))
        
        if len(remaining_demo_indices) >= args.val_min_episodes:
            val_demo_indices = remaining_demo_indices
            print(f"Using {len(val_demo_indices)} remaining episodes for validation")
        else:
            val_size = max(args.val_min_episodes, int(len(train_demo_indices) * args.val_split))
            rng = np.random.RandomState(args.seed)
            val_demo_indices = sorted(rng.choice(train_demo_indices, size=val_size, replace=False).tolist())
            print(f"Randomly selected {len(val_demo_indices)} episodes from training for validation")
        
        try:
            min_pred_horizon = max(args.obs_horizon, 2)  # At least obs_horizon, minimum 2
            val_dataset = SoftGymSegmentedDatasetSimple(
                dataset_path=dataset_path,
                pred_horizon=min_pred_horizon,  # Must be >= obs_horizon to get enough observations
                obs_horizon=args.obs_horizon,
                action_horizon=1,
                demo_indices=val_demo_indices,
                use_gripper_segmentation=True,  # Use gripper-based segmentation
                use_contact_segmentation=False,
                min_segment_length=args.min_segment_length,
            )
            print(f"Created validation dataset with {len(val_dataset)} samples")
        except Exception as e:
            print(f"Warning: Could not create validation dataset: {e}")
            val_dataset = None

    # save training data statistics
    stats = getattr(dataset, "stats", None)

    # create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['obs'].shape:", batch['obs'].shape)
    print("batch['reference_pos'].shape", batch['reference_pos'].shape)

    # create Position2DFlowDecoder model
    obs_cond_dim = args.obs_horizon * args.obs_dim
    position_decoder = Position2DFlowDecoder(
        obs_cond_dim=obs_cond_dim,
        sigma=args.sigma,
        fm_timesteps=args.position_decoder_fm_timesteps,
        down_dims=args.position_decoder_down_dims,
        n_groups=args.position_decoder_n_groups,
        backbone=args.backbone,
        mlp_hidden_dims=args.mlp_hidden_dims,
        mlp_activation=args.mlp_activation,
        use_flow_matching=use_flow_matching,
        num_pickers=args.num_pickers,  # Pass num_pickers to model
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in position_decoder.parameters())
    trainable_params = sum(p.numel() for p in position_decoder.parameters() if p.requires_grad)
    print(f"Position decoder backbone: {args.backbone}")
    print(f"Using flow matching: {use_flow_matching}")
    print(f"Total parameters: {total_params:.2e}")
    print(f"Trainable parameters: {trainable_params:.2e}")

    num_epochs = args.epochs

    # EMA
    ema = EMAModel(parameters=position_decoder.parameters(), power=args.ema_power)

    optimizer = torch.optim.AdamW(
        params=position_decoder.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    if wandb_available:
        try:
            wandb.watch(position_decoder, log="all", log_freq=200)
        except Exception:
            pass

    # Create checkpoint directory
    ckpt_dir = f"{args.output_dir}/{exp_name}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            epoch_position_loss = list()
            
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    nobs = nbatch['obs'].to(device)
                    reference_pos = nbatch['reference_pos'].to(device)

                    # Add noise to observations for regularization (only during training)
                    if args.obs_noise_scale > 0.0:
                        if args.obs_noise_type == "gaussian":
                            # Gaussian noise: N(0, obs_noise_scale^2)
                            noise = torch.randn_like(nobs) * args.obs_noise_scale
                        elif args.obs_noise_type == "uniform":
                            # Uniform noise: U(-obs_noise_scale, obs_noise_scale)
                            noise = (torch.rand_like(nobs) * 2 - 1) * args.obs_noise_scale
                        else:
                            raise ValueError(f"Unknown noise type: {args.obs_noise_type}")
                        nobs = nobs + noise
                    
                    # Flatten observations for conditioning
                    obs_cond = nobs.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

                    # Compute position loss
                    position_loss, position_loss_dict = position_decoder.compute_loss(
                        obs_cond=obs_cond,
                        gt_positions=reference_pos,
                        x_0=None
                    )

                    position_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    ema.step(position_decoder.parameters())

                    loss_cpu = position_loss.item()
                    epoch_loss.append(loss_cpu)
                    epoch_position_loss.append(loss_cpu)
                    
                    tepoch.set_postfix(loss=loss_cpu)

            # Log at epoch level
            mean_epoch_loss = np.mean(epoch_loss) if epoch_loss else 0.0
            mean_position_loss = np.mean(epoch_position_loss) if epoch_position_loss else 0.0
            
            tglobal.set_postfix(loss=mean_epoch_loss)
            
            if wandb_available:
                log_dict = {
                    "train/epoch_loss": mean_epoch_loss,
                    "train/position_loss": mean_position_loss,
                    "train/lr": lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else None,
                    "train/epoch": epoch_idx
                }
                wandb.log(log_dict)
            
            # Evaluation step - visualize on both training and validation datasets
            if args.eval_every > 0 and (epoch_idx + 1) % args.eval_every == 0:
                # Create EMA model for evaluation
                ema_model = Position2DFlowDecoder(
                    obs_cond_dim=obs_cond_dim,
                    sigma=args.sigma,
                    fm_timesteps=args.position_decoder_fm_timesteps,
                    down_dims=args.position_decoder_down_dims,
                    n_groups=args.position_decoder_n_groups,
                    backbone=args.backbone,
                    mlp_hidden_dims=args.mlp_hidden_dims,
                    mlp_activation=args.mlp_activation,
                    use_flow_matching=use_flow_matching,
                    num_pickers=args.num_pickers,  # Pass num_pickers to model
                ).to(device)
                
                ema.copy_to(ema_model.parameters())
                ema_model.eval()
                print(f"Running evaluation at epoch {epoch_idx + 1}...")
                
                # Prepare random number generator
                if args.eval_sample_seed is not None:
                    rng = np.random.RandomState(args.eval_sample_seed + epoch_idx)
                else:
                    rng = np.random.RandomState(args.seed + epoch_idx)
                
                eval_log_dict = {"eval/epoch": epoch_idx + 1}
                
                # Evaluate on training dataset
                try:
                    train_sample_idx = rng.randint(0, len(dataset))
                    print(f"  Evaluating on training dataset (sample {train_sample_idx})...")
                    
                    train_vis_results = visualize_position_predictions(
                        model=ema_model,
                        dataset=dataset,
                        stats=stats,
                        sample_idx=train_sample_idx,
                        out_path=f"{args.output_dir}/{exp_name}/position_pred_train_epoch_{epoch_idx + 1}.png",
                        device=device,
                    )
                    print(f"  Saved training visualization to: {train_vis_results['image_path']}")
                    
                    eval_log_dict["eval/train_pred_error"] = train_vis_results.get('pred_error', 0.0)
                    if wandb_available:
                        eval_log_dict["eval/train_position_prediction"] = wandb.Image(train_vis_results['image_path'])
                except Exception as e:
                    print(f"  Warning: Failed to visualize on training dataset: {e}")
                
                # Evaluate on validation dataset (if available)
                if val_dataset is not None and len(val_dataset) > 0:
                    try:
                        val_sample_idx = rng.randint(0, len(val_dataset))
                        print(f"  Evaluating on validation dataset (sample {val_sample_idx})...")
                        
                        val_vis_results = visualize_position_predictions(
                            model=ema_model,
                            dataset=val_dataset,
                            stats=stats,
                            sample_idx=val_sample_idx,
                            out_path=f"{args.output_dir}/{exp_name}/position_pred_val_epoch_{epoch_idx + 1}.png",
                            device=device,
                        )
                        print(f"  Saved validation visualization to: {val_vis_results['image_path']}")
                        
                        eval_log_dict["eval/val_pred_error"] = val_vis_results.get('pred_error', 0.0)
                        if wandb_available:
                            eval_log_dict["eval/val_position_prediction"] = wandb.Image(val_vis_results['image_path'])
                    except Exception as e:
                        print(f"  Warning: Failed to visualize on validation dataset: {e}")
                
                # Log to wandb
                if wandb_available and eval_log_dict:
                    wandb.log(eval_log_dict)

    # Copy EMA weights and save checkpoint
    ema_model = Position2DFlowDecoder(
        obs_cond_dim=obs_cond_dim,
        sigma=args.sigma,
        fm_timesteps=args.position_decoder_fm_timesteps,
        down_dims=args.position_decoder_down_dims,
        n_groups=args.position_decoder_n_groups,
        backbone=args.backbone,
        mlp_hidden_dims=args.mlp_hidden_dims,
        mlp_activation=args.mlp_activation,
        use_flow_matching=use_flow_matching,
        num_pickers=args.num_pickers,  # Pass num_pickers to model
    ).to(device)
    
    ema.copy_to(ema_model.parameters())
    
    # save final checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"position_decoder.pt")
    torch.save(ema_model.state_dict(), ckpt_path)
    print(f"Saved final checkpoint: {ckpt_path}")

    if wandb_available:
        try:
            wandb.save(ckpt_path)
            wandb.finish()
        except Exception:
            pass

