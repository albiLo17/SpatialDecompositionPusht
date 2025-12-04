"""Diffusion model class for action prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

from .conditional_unet import ConditionalUnet1D


class Diffusion(nn.Module):
    """
    Diffusion model for action prediction using DDPM.
    
    This class encapsulates the diffusion process including:
    - Noise prediction network (ConditionalUnet1D)
    - DDPM scheduler
    - Training loss computation
    - Action sampling/inference
    """
    
    def __init__(
        self,
        act_dim: int,
        obs_horizon: int,
        act_horizon: int,
        pred_horizon: int,
        obs_dim: int,
        num_diffusion_iters: int = 100,
        diffusion_step_embed_dim: int = 256,
        down_dims: list = None,
        n_groups: int = 8,
        beta_schedule: str = 'squaredcos_cap_v2',
        clip_sample: bool = True,
        prediction_type: str = 'epsilon'
    ):
        """
        Args:
            act_dim: Action dimension
            obs_horizon: Observation horizon
            act_horizon: Action horizon
            pred_horizon: Prediction horizon
            obs_dim: Observation dimension
            num_diffusion_iters: Number of diffusion iterations
            diffusion_step_embed_dim: Dimension of diffusion step embedding
            down_dims: Channel dimensions for U-Net downsampling
            n_groups: Number of groups for GroupNorm
            beta_schedule: Beta schedule for noise scheduler
            clip_sample: Whether to clip samples to [-1, 1]
            prediction_type: Type of prediction ('epsilon' for noise prediction)
        """
        super().__init__()
        
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.num_diffusion_iters = num_diffusion_iters
        
        if down_dims is None:
            down_dims = [256, 512, 1024]
        
        # Noise prediction network
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=obs_horizon * obs_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            n_groups=n_groups,
        )
        
        # DDPM scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            prediction_type=prediction_type
        )
    
    def compute_loss(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffusion training loss.
        
        Args:
            obs_seq: Observation sequence, shape (B, obs_horizon, obs_dim)
            action_seq: Action sequence, shape (B, pred_horizon, act_dim)
            
        Returns:
            Loss value
        """
        B = obs_seq.shape[0]
        
        # Observation as FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
        
        # Sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)
        
        # Sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=obs_seq.device
        ).long()
        
        # Add noise to the clean actions according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(
            action_seq, noise, timesteps)
        
        # Predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond)
        
        return F.mse_loss(noise_pred, noise)
    
    def get_action(
        self, 
        obs_seq: torch.Tensor, 
        return_chain: bool = False, 
        grad: bool = False
    ) -> torch.Tensor:
        """
        Sample actions from the diffusion model.
        
        Args:
            obs_seq: Observation sequence, shape (B, obs_horizon, obs_dim)
            return_chain: Whether to return the denoising chain
            grad: Whether to enable gradients (for differentiable sampling)
            
        Returns:
            Actions, shape (B, act_horizon, act_dim)
            If return_chain=True, also returns the denoising chain
        """
        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs_seq.shape[0]
        
        # Only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        
        if grad:
            return self.sample_sequence(obs_seq, start, end, B, return_chain)
        with torch.no_grad():
            return self.sample_sequence(obs_seq, start, end, B, return_chain)
    
    def sample_sequence(
        self, 
        obs_seq: torch.Tensor, 
        start: int, 
        end: int, 
        B: int, 
        return_chain: bool
    ) -> torch.Tensor:
        """
        Sample a sequence of actions using the diffusion process.
        
        Args:
            obs_seq: Observation sequence
            start: Start index for action extraction
            end: End index for action extraction
            B: Batch size
            return_chain: Whether to return the denoising chain
            
        Returns:
            Actions, shape (B, act_horizon, act_dim)
            If return_chain=True, also returns the denoising chain
        """
        chain = []
        obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
        
        # Initialize action from Gaussian noise
        noisy_action_seq = torch.randn(
            (B, self.pred_horizon, self.act_dim), device=obs_seq.device)
        chain.append(noisy_action_seq[:, start:end])
        
        # Set timesteps for inference
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        
        for k in self.noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self.noise_pred_net(
                sample=noisy_action_seq,
                timestep=k,
                global_cond=obs_cond,
            )
            
            # Inverse diffusion step (remove noise)
            noisy_action_seq = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action_seq,
            ).prev_sample
            
            if return_chain:
                chain.append(noisy_action_seq[:, start:end])
        
        if return_chain:
            return noisy_action_seq[:, start:end], chain
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)

