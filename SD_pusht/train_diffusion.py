import argparse
import os
import math
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from eval_diffusion import evaluate_model
from network import ConditionalUnet1D
import gdown
from push_t_dataset import PushTStateDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", type=str, default="datasets/pusht_cchi_v7_replay.zarr.zip")
    p.add_argument("--max_demos", type=int, default=None,
                   help="Maximum number of demonstration samples (dataset entries) to use for training")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default="sd-pusht")
    # add evaluation step every N epochs
    p.add_argument("--eval-every", type=int, default=10, help="Evaluate every N epochs (0 = no evaluation during training)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    exp_name = f"sd-pusht-dp-demos-{args.max_demos}-seed{args.seed}"
    print(f"Experiment: {exp_name}")

    # optional wandb
    if args.wandb:
        print("Using Weights & Biases logging")
        import wandb
        wandb_available = True
        wandb.init(project=args.wandb_project, config=vars(args))
        # define also name of the run
        wandb.run.name = exp_name
    else:
        wandb_available = False

    # download demonstration data from Google Drive if missing
    dataset_path = args.dataset_path
    if not os.path.isfile(dataset_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)

    # parameters (kept from original defaults)
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8

    # create dataset from file
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        max_demos=args.max_demos
    )


    # save training data statistics (min, max) for each dim (if original dataset exposes .stats)
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

    # observation and action dimensions corresponding to the output of PushTEnv
    obs_dim = 5
    action_dim = 2

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # diffusion scheduler
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # device transfer
    noise_pred_net = noise_pred_net.to(device)

    num_epochs = args.epochs

    # EMA
    ema = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)

    optimizer = torch.optim.AdamW(params=noise_pred_net.parameters(), lr=1e-4, weight_decay=1e-6)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    if wandb_available:
        try:
            wandb.watch(noise_pred_net, log="all", log_freq=200)
        except Exception:
            pass

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)
                    B = nobs.shape[0]

                    obs_cond = nobs[:, :obs_horizon, :]
                    obs_cond = obs_cond.flatten(start_dim=1)

                    noise = torch.randn(naction.shape, device=device)

                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)

                    loss = nn.functional.mse_loss(noise_pred, noise)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    ema.step(noise_pred_net.parameters())

                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

                    if wandb_available:
                        wandb.log({
                            "train/batch_loss": loss_cpu,
                            "train/lr": lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else None,
                            "train/epoch": epoch_idx
                        })

            mean_epoch_loss = np.mean(epoch_loss) if epoch_loss else 0.0
            tglobal.set_postfix(loss=mean_epoch_loss)
            if wandb_available:
                wandb.log({"train/epoch_loss": mean_epoch_loss, "train/epoch": epoch_idx})
            # evaluation step
            if args.eval_every > 0 and (epoch_idx + 1) % args.eval_every == 0:
                
                # Evaluate EMA-weighted model without modifying the training model
                ema_model = ConditionalUnet1D(
                    input_dim=action_dim,
                    global_cond_dim=obs_dim * obs_horizon
                ).to(device)
                ema.copy_to(ema_model.parameters())   # copy averaged weights into a fresh model
                ema_model.eval()
                print(f"Running evaluation at epoch {epoch_idx + 1}...")
                eval_results = evaluate_model(
                    noise_pred_net=ema_model,
                    noise_scheduler=noise_scheduler,
                    stats=stats,
                    out_path=f"log/dp/{exp_name}/eval_epoch_{epoch_idx + 1}.mp4",
                    num_envs=16,
                    max_steps=300,
                    num_diffusion_iters=num_diffusion_iters,
                    pred_horizon=pred_horizon,
                    obs_horizon=obs_horizon,
                    action_horizon=action_horizon,
                    device=device
                )
                
                # log success rate in wandb
                if wandb_available:
                    wandb.log({"eval/epoch": epoch_idx + 1, 
                               "eval/Success Rate": eval_results['success_rate']})

    # Copy EMA weights to a fresh model and save checkpoint (do not overwrite training model)
    ema_model = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon).to(device)
    ema.copy_to(ema_model.parameters())
    # save checkpoint
    ckpt_dir = f"log/dp/{exp_name}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ema_noise_pred_net.pt")
    torch.save(ema_model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    if wandb_available:
        try:
            wandb.save(ckpt_path)
            wandb.finish()
        except Exception:
            pass