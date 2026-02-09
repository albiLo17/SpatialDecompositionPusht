#!/usr/bin/env python3
"""Training script for diffusion/flow matching policy on PushT."""

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

from SD_pusht.models import Diffusion, FlowMatching
from SD_pusht.datasets import PushTStateDataset
from SD_pusht.utils.evaluation import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion/flow matching policy on PushT")
    parser.add_argument("--use-flow-matching", action="store_true",
                       help="Use flow matching instead of diffusion")
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
    parser.add_argument("--wandb-project", type=str, default="sd-pusht",
                       help="WandB project name")
    parser.add_argument("--eval-every", type=int, default=10,
                       help="Evaluate every N epochs (0 = no evaluation during training)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--pred-horizon", type=int, default=16,
                       help="Prediction horizon")
    parser.add_argument("--obs-horizon", type=int, default=2,
                       help="Observation horizon")
    parser.add_argument("--action-horizon", type=int, default=8,
                       help="Action horizon")
    parser.add_argument("--obs-dim", type=int, default=5,
                       help="Observation dimension")
    parser.add_argument("--action-dim", type=int, default=2,
                       help="Action dimension")
    parser.add_argument("--num-diffusion-iters", type=int, default=100,
                       help="Number of diffusion iterations (for diffusion model)")
    parser.add_argument("--fm-timesteps", type=int, default=100,
                       help="Number of timesteps for flow matching inference")
    parser.add_argument("--sigma", type=float, default=0.0,
                       help="Noise scale for flow matching paths")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-6,
                       help="Weight decay")
    parser.add_argument("--ema-power", type=float, default=0.75,
                       help="EMA power parameter")
    parser.add_argument("--output-dir", type=str, default="log/dp",
                       help="Output directory for checkpoints and logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create experiment name based on model type
    model_type = "fm" if args.use_flow_matching else "diffusion"
    exp_name = f"sd-pusht-{model_type}-demos-{args.max_demos}-seed{args.seed}"
    print(f"Experiment: {exp_name}")
    print(f"Using model: {'Flow Matching' if args.use_flow_matching else 'Diffusion'}")
    
    # Create experiment directory and save config
    exp_dir = f"{args.output_dir}/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.json")
    
    # Convert args to dict and save as JSON
    config_dict = vars(args).copy()
    # Convert any non-serializable objects to strings
    for key, value in config_dict.items():
        if not isinstance(value, (str, int, float, bool, type(None))):
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

    # create dataset from file
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        max_demos=args.max_demos
    )

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
    print("batch['action'].shape", batch['action'].shape)

    # create model (Diffusion or FlowMatching)
    if args.use_flow_matching:
        agent = FlowMatching(
            act_dim=args.action_dim,
            obs_horizon=args.obs_horizon,
            act_horizon=args.action_horizon,
            pred_horizon=args.pred_horizon,
            obs_dim=args.obs_dim,
            sigma=args.sigma,
            fm_timesteps=args.fm_timesteps
        ).to(device)
    else:
        agent = Diffusion(
            act_dim=args.action_dim,
            obs_horizon=args.obs_horizon,
            act_horizon=args.action_horizon,
            pred_horizon=args.pred_horizon,
            obs_dim=args.obs_dim,
            num_diffusion_iters=args.num_diffusion_iters
        ).to(device)

    num_epochs = args.epochs

    # EMA
    ema = EMAModel(parameters=agent.parameters(), power=args.ema_power)

    optimizer = torch.optim.AdamW(
        params=agent.parameters(), 
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
            wandb.watch(agent, log="all", log_freq=200)
        except Exception:
            pass

    # Track best evaluation metrics
    best_success_rate = -1.0
    best_epoch = -1
    best_model_path = None
    
    # Create checkpoint directory
    ckpt_dir = f"{args.output_dir}/{exp_name}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)

                    # Compute loss using the model's compute_loss method
                    loss = agent.compute_loss(obs_seq=nobs, action_seq=naction)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    ema.step(agent.parameters())

                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            # Log at epoch level instead of batch level
            mean_epoch_loss = np.mean(epoch_loss) if epoch_loss else 0.0
            tglobal.set_postfix(loss=mean_epoch_loss)
            
            if wandb_available:
                wandb.log({
                    "train/epoch_loss": mean_epoch_loss,
                    "train/lr": lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else None,
                    "train/epoch": epoch_idx
                })
            
            # evaluation step
            if args.eval_every > 0 and (epoch_idx + 1) % args.eval_every == 0:
                # Create EMA model for evaluation
                if args.use_flow_matching:
                    ema_model = FlowMatching(
                        act_dim=args.action_dim,
                        obs_horizon=args.obs_horizon,
                        act_horizon=args.action_horizon,
                        pred_horizon=args.pred_horizon,
                        obs_dim=args.obs_dim,
                        sigma=args.sigma,
                        fm_timesteps=args.fm_timesteps
                    ).to(device)
                else:
                    ema_model = Diffusion(
                        act_dim=args.action_dim,
                        obs_horizon=args.obs_horizon,
                        act_horizon=args.action_horizon,
                        pred_horizon=args.pred_horizon,
                        obs_dim=args.obs_dim,
                        num_diffusion_iters=args.num_diffusion_iters
                    ).to(device)
                
                ema.copy_to(ema_model.parameters())
                ema_model.eval()
                print(f"Running evaluation at epoch {epoch_idx + 1}...")
                
                # Evaluate using the model's get_action method
                eval_results = evaluate_model(
                    model=ema_model,
                    stats=stats,
                    out_path=f"{args.output_dir}/{exp_name}/eval_epoch_{epoch_idx + 1}.mp4",
                    num_envs=64,
                    max_steps=300,
                    pred_horizon=args.pred_horizon,
                    obs_horizon=args.obs_horizon,
                    action_horizon=args.action_horizon,
                    device=device,
                    use_flow_matching=args.use_flow_matching
                )
                
                # Check if this is the best model so far
                current_success_rate = eval_results['success_rate']
                if current_success_rate > best_success_rate:
                    best_success_rate = current_success_rate
                    best_epoch = epoch_idx + 1
                    
                    # Save best model
                    best_ckpt_path = os.path.join(ckpt_dir, f"best_ema_model.pt")
                    torch.save(ema_model.state_dict(), best_ckpt_path)
                    best_model_path = best_ckpt_path
                    print(f"New best model! Success rate: {best_success_rate:.4f} at epoch {best_epoch}")
                    print(f"Saved best model to: {best_ckpt_path}")
                
                # log success rate and video in wandb
                if wandb_available:
                    log_dict = {
                        "eval/epoch": epoch_idx + 1, 
                        "eval/Success Rate": eval_results['success_rate'],
                        "eval/mean_score": eval_results['mean_score'],
                        "eval/max_score": eval_results['max_score'],  # Max cumulative reward
                        "eval/mean_max_single_reward": eval_results['mean_max_single_reward'],  # Average of max single-step reward per environment
                        "eval/best_success_rate": best_success_rate,
                        "eval/best_epoch": best_epoch,
                    }
                    # Log video if available - use file path instead of raw data to avoid moviepy dependency
                    if eval_results.get('video_path') is not None:
                        video_path = eval_results['video_path']
                        if os.path.exists(video_path):
                            log_dict["eval/video"] = wandb.Video(video_path, fps=30, format="mp4")
                    wandb.log(log_dict)

    # Copy EMA weights to a fresh model and save checkpoint
    if args.use_flow_matching:
        ema_model = FlowMatching(
            act_dim=args.action_dim,
            obs_horizon=args.obs_horizon,
            act_horizon=args.action_horizon,
            pred_horizon=args.pred_horizon,
            obs_dim=args.obs_dim,
            sigma=args.sigma,
            fm_timesteps=args.fm_timesteps
        ).to(device)
    else:
        ema_model = Diffusion(
            act_dim=args.action_dim,
            obs_horizon=args.obs_horizon,
            act_horizon=args.action_horizon,
            pred_horizon=args.pred_horizon,
            obs_dim=args.obs_dim,
            num_diffusion_iters=args.num_diffusion_iters
        ).to(device)
    
    ema.copy_to(ema_model.parameters())
    
    # save final checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"ema_model.pt")
    torch.save(ema_model.state_dict(), ckpt_path)
    print(f"Saved final checkpoint: {ckpt_path}")
    
    # Print summary of best model
    if best_model_path is not None:
        print(f"\n{'='*50}")
        print("Best Model Summary:")
        print(f"{'='*50}")
        print(f"Best Success Rate: {best_success_rate:.4f}")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Model Path: {best_model_path}")
        print(f"{'='*50}")
    else:
        print("Note: No evaluations were performed, so no best model was saved.")

    if wandb_available:
        try:
            wandb.save(ckpt_path)
            wandb.finish()
        except Exception:
            pass
