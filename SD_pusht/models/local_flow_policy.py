"""Local Flow Matching model with frame decoder for spatial decomposition."""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .conditional_unet import ConditionalUnet1D
from .conditional_flow_matcher import ConditionalFlowMatcher
from .flow_matching import FlowMatching
from ..utils.transformations import (
    to_transformation_matrix_th,
    rotvec_to_rotmat,
    rotmat_to_rotvec,
    transform_flat_pose_vector_th,
    special_gramschmidt,
)


class PoseFlowDecoder(nn.Module):
    """
    Flow matching decoder for predicting a single pose (translation + rotation).
    
    This is a simplified version adapted from SmallPoseFlowModel to work with
    the ConditionalFlowMatcher interface.
    """
    
    def __init__(
        self,
        obs_cond_dim: int,
        sigma: float = 0.0,
        fm_timesteps: int = 8,
        diffusion_step_embed_dim: int = 256,
        down_dims: list = None,
        n_groups: int = 4,
    ):
        """
        Args:
            obs_cond_dim: Dimension of observation conditioning
            sigma: Noise scale for flow matching paths
            fm_timesteps: Number of timesteps for flow matching inference
            diffusion_step_embed_dim: Dimension of timestep embedding
            down_dims: Channel dimensions for U-Net downsampling
            n_groups: Number of groups for GroupNorm
        """
        super().__init__()
        
        self.POSE_DIM = 9  # 3 for position, 6 for rotation matrix (2 vectors)
        self.fm_timesteps = fm_timesteps
        
        if down_dims is None:
            down_dims = [128, 128, 128]
        
        # Flow matching network for pose prediction
        # Note: We use kernel_size=1 since we're predicting a single pose (horizon=1)
        self.pose_net = ConditionalUnet1D(
            input_dim=self.POSE_DIM,
            global_cond_dim=obs_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=1,  # Single timestep
            n_groups=n_groups,
        )
        
        # Conditional Flow Matcher
        self.FM = ConditionalFlowMatcher(sigma=sigma)
    
    def forward(
        self,
        obs_cond: torch.Tensor,
        x_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict pose (translation + rotation) from observation conditioning.
        
        Args:
            obs_cond: Observation conditioning, shape (B, obs_cond_dim)
            x_init: Optional initial noise, shape (B, 1, POSE_DIM). If None, sampled randomly.
        
        Returns:
            xyz: Translation, shape (B, 3)
            rot: Rotation matrix, shape (B, 3, 3)
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        with torch.no_grad():
            # Initialize from noise (single timestep, so horizon=1)
            if x_init is None:
                noisy_pose = torch.randn(
                    (B, 1, self.POSE_DIM), device=device
                )
            else:
                noisy_pose = x_init
                assert noisy_pose.shape == (B, 1, self.POSE_DIM), \
                    f"Expected x_init shape ({B}, 1, {self.POSE_DIM}), got {noisy_pose.shape}"
            
            # Integrate the flow over timesteps
            for i in range(self.fm_timesteps):
                timestep = torch.tensor(
                    [i / self.fm_timesteps], device=device
                ).repeat(B,)
                
                if i == 0:
                    vt = self.pose_net(
                        noisy_pose, timestep, global_cond=obs_cond
                    )
                    denoised_pose = (vt * 1 / self.fm_timesteps + noisy_pose)
                else:
                    vt = self.pose_net(
                        denoised_pose, timestep, global_cond=obs_cond
                    )
                    denoised_pose = (vt * 1 / self.fm_timesteps + denoised_pose)
        
        # Extract translation and rotation vectors
        pose_flat = denoised_pose.squeeze(1)  # (B, POSE_DIM)
        xyz = pose_flat[:, :3]
        w1 = pose_flat[:, 3:6]
        w2 = pose_flat[:, 6:9]
        
        # Apply orthonormalization to form valid rotation matrix
        rot = special_gramschmidt(w1, w2)
        
        return xyz, rot
    
    def compute_loss(
        self,
        obs_cond: torch.Tensor,
        gt_poses: torch.Tensor,
        trans_coeff: float = 1.0,
        rot_coeff: float = 1.0,
        x_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute flow matching loss for pose prediction.
        
        Args:
            obs_cond: Observation conditioning, shape (B, obs_cond_dim)
            gt_poses: Ground truth poses as 4x4 transformation matrices, shape (B, 4, 4)
            trans_coeff: Coefficient for translation loss
            rot_coeff: Coefficient for rotation loss
            x_0: Optional pre-sampled noise, shape (B, 1, POSE_DIM)
        
        Returns:
            total_loss: Total loss value
            loss_dict: Dictionary with individual loss components
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        # Convert GT poses to flat representation (xyz + 2 rotation vectors)
        gt_xyz = gt_poses[:, :3, 3]  # (B, 3)
        gt_w1 = gt_poses[:, :3, 0]   # First column
        gt_w2 = gt_poses[:, :3, 1]   # Second column
        gt_pose_flat = torch.cat([gt_xyz, gt_w1, gt_w2], dim=-1)  # (B, 9)
        gt_pose_flat = gt_pose_flat.unsqueeze(1)  # (B, 1, 9) - add horizon dimension
        
        # Sample noise
        if x_0 is None:
            x0 = torch.randn(gt_pose_flat.shape, device=device)
        else:
            x0 = x_0
            assert x0.shape == gt_pose_flat.shape, \
                f"Expected x_0 shape {gt_pose_flat.shape}, got {x0.shape}"
        
        # Sample location along path and get conditional flow
        timestep, xt, ut = self.FM.sample_location_and_conditional_flow(x0, gt_pose_flat)
        
        # Predict velocity
        vt = self.pose_net(xt, timestep, global_cond=obs_cond)
        
        # Compute losses for translation and rotation separately
        loss_trans = F.mse_loss(vt[:, :, :3], ut[:, :, :3])
        loss_rot = F.mse_loss(vt[:, :, 3:], ut[:, :, 3:])
        
        total_loss = trans_coeff * loss_trans + rot_coeff * loss_rot
        
        loss_dict = {
            "trans": loss_trans.item(),
            "rot": loss_rot.item(),
            "total": total_loss.item(),
        }
        
        return total_loss, loss_dict


class LocalFlowPolicy(FlowMatching):
    """
    Local Flow Matching policy with frame decoder for spatial decomposition.
    
    This extends FlowMatching to predict actions in a local frame, then transform
    them to the world frame using a predicted reference frame.
    """
    
    def __init__(
        self,
        act_dim: int,
        obs_horizon: int,
        act_horizon: int,
        pred_horizon: int,
        obs_dim: int,
        sigma: float = 0.0,
        fm_timesteps: int = 100,
        diffusion_step_embed_dim: int = 256,
        down_dims: list = None,
        n_groups: int = 8,
        # Frame decoder parameters
        use_frame_decoder: bool = True,
        frame_decoder_down_dims: list = None,
        frame_decoder_n_groups: int = 4,
        frame_decoder_fm_timesteps: int = 8,
        # Loss coefficients
        frame_trans_coeff: float = 1.0,
        frame_rot_coeff: float = 1.0,
        frame_loss_coeff: float = 1.0,
        # Noise sharing
        share_noise: bool = False,
        shared_noise_base: str = "action",  # "action", "frame", or "combinatory"
    ):
        """
        Args:
            act_dim: Action dimension
            obs_horizon: Observation horizon
            act_horizon: Action horizon
            pred_horizon: Prediction horizon
            obs_dim: Observation dimension
            sigma: Noise scale for flow matching paths
            fm_timesteps: Number of timesteps for flow matching inference
            diffusion_step_embed_dim: Dimension of timestep embedding
            down_dims: Channel dimensions for U-Net downsampling
            n_groups: Number of groups for GroupNorm
            use_frame_decoder: Whether to use frame decoder
            frame_decoder_down_dims: Channel dimensions for frame decoder U-Net
            frame_decoder_n_groups: Number of groups for frame decoder GroupNorm
            frame_decoder_fm_timesteps: Number of timesteps for frame decoder inference
            frame_trans_coeff: Coefficient for frame translation loss
            frame_rot_coeff: Coefficient for frame rotation loss
            frame_loss_coeff: Overall coefficient for frame loss
            share_noise: Whether to share noise between frame and action predictions
            shared_noise_base: Base for noise sharing ("action", "frame", or "combinatory")
        """
        super().__init__(
            act_dim=act_dim,
            obs_horizon=obs_horizon,
            act_horizon=act_horizon,
            pred_horizon=pred_horizon,
            obs_dim=obs_dim,
            sigma=sigma,
            fm_timesteps=fm_timesteps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            n_groups=n_groups,
        )
        
        self.use_frame_decoder = use_frame_decoder
        self.share_noise = share_noise
        self.shared_noise_base = shared_noise_base
        self.frame_loss_coeff = frame_loss_coeff
        
        if use_frame_decoder:
            obs_cond_dim = obs_horizon * obs_dim
            self.frame_decoder = PoseFlowDecoder(
                obs_cond_dim=obs_cond_dim,
                sigma=sigma,
                fm_timesteps=frame_decoder_fm_timesteps,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=frame_decoder_down_dims,
                n_groups=frame_decoder_n_groups,
            )
            self.frame_trans_coeff = frame_trans_coeff
            self.frame_rot_coeff = frame_rot_coeff
        else:
            self.frame_decoder = None
    
    def _sample_aligned_noise(
        self,
        B: int,
        device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sample aligned noise for action and frame predictions.
        
        Args:
            B: Batch size
            device: Device to sample on
        
        Returns:
            action_noise: Noise for action prediction, shape (B, pred_horizon, act_dim) or None
            frame_noise: Noise for frame prediction, shape (B, 1, 9) or None
        """
        if not self.share_noise:
            return None, None
        
        H = self.pred_horizon
        F = 9  # POSE_DIM
        A = self.act_dim
        
        if self.shared_noise_base == "action":
            # Sample action noise first, then project to frame noise
            action_noise = torch.randn((B, H, A), device=device)
            # Aggregate along horizon dimension
            frame_noise = action_noise.mean(dim=1, keepdim=True)  # (B, 1, A)
            variance_rescale = np.sqrt(H)
            frame_noise = frame_noise * variance_rescale
            # Project from action_dim to 9D (simple padding/truncation for now)
            if A >= 9:
                frame_noise = frame_noise[:, :, :9]
            else:
                padding = torch.zeros((B, 1, 9 - A), device=device)
                frame_noise = torch.cat([frame_noise, padding], dim=-1)
        
        elif self.shared_noise_base == "frame":
            # Sample frame noise first, then project to action noise
            frame_noise = torch.randn((B, 1, F), device=device)
            # Expand to action noise
            action_noise = frame_noise.repeat(1, H, 1)
            variance_rescale = np.sqrt(H)
            action_noise = action_noise / variance_rescale
            # Project from 9D to action_dim
            if A <= 9:
                action_noise = action_noise[:, :, :A]
            else:
                padding = torch.randn((B, H, A - 9), device=device) / variance_rescale
                action_noise = torch.cat([action_noise, padding], dim=-1)
        
        elif self.shared_noise_base == "combinatory":
            # Sample from a higher-dimensional space and project
            full_noise = torch.randn((B, H, F * A), device=device)
            # Simple projection: take mean for frame, reshape for action
            frame_noise = full_noise.mean(dim=1, keepdim=True) * np.sqrt(H)
            frame_noise = frame_noise[:, :, :F]
            action_noise = full_noise[:, :, :A]
        else:
            raise ValueError(f"Unknown shared_noise_base: {self.shared_noise_base}")
        
        return action_noise, frame_noise
    
    def get_action(
        self,
        obs_seq: torch.Tensor,
        reference_frame: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample actions from the local flow matching model.
        
        Args:
            obs_seq: Observation sequence, shape (B, obs_horizon, obs_dim)
            reference_frame: Optional reference frame for training, shape (B, 4, 4)
        
        Returns:
            Dictionary with:
                - "actions": World frame actions, shape (B, act_horizon, act_dim)
                - "base_T": Predicted base frame, shape (B, 4, 4)
                - "local_actions": Local frame actions, shape (B, act_horizon, act_dim)
        """
        B = obs_seq.shape[0]
        device = obs_seq.device
        obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
        
        with torch.no_grad():
            # Predict global frame if using frame decoder
            if self.use_frame_decoder:
                if self.share_noise:
                    _, frame_noise = self._sample_aligned_noise(B, device)
                else:
                    frame_noise = None
                
                global_xyz, global_rot = self.frame_decoder(obs_cond, x_init=frame_noise)
                global_frame = to_transformation_matrix_th(t=global_xyz, R=global_rot)
            else:
                # Use identity or provided reference frame
                if reference_frame is not None:
                    global_frame = reference_frame
                else:
                    # Identity frame
                    global_frame = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
            
            # Sample aligned noise for actions if sharing
            if self.share_noise:
                action_noise, _ = self._sample_aligned_noise(B, device)
            else:
                action_noise = None
            
            # Get local actions using parent's flow matching
            # We need to override the noise sampling in get_action
            # For now, we'll manually do the flow integration with custom noise
            if action_noise is not None:
                # Use custom noise initialization
                noisy_action_seq = action_noise  # (B, pred_horizon, act_dim)
            else:
                noisy_action_seq = torch.randn(
                    (B, self.pred_horizon, self.act_dim), device=device
                )
            
            # Integrate the flow over timesteps
            for i in range(self.fm_timesteps):
                timestep = torch.tensor(
                    [i / self.fm_timesteps], device=device
                ).repeat(B,)
                
                if i == 0:
                    vt = self.noise_pred_net(
                        noisy_action_seq, timestep, global_cond=obs_cond
                    )
                    denoised_action_seq = (vt * 1 / self.fm_timesteps + noisy_action_seq)
                else:
                    vt = self.noise_pred_net(
                        denoised_action_seq, timestep, global_cond=obs_cond
                    )
                    denoised_action_seq = (vt * 1 / self.fm_timesteps + denoised_action_seq)
            
            # Extract local actions (act_horizon portion)
            start = self.obs_horizon - 1
            end = start + self.act_horizon
            local_actions = denoised_action_seq[:, start:end]  # (B, act_horizon, act_dim)
            
            # Transform local actions to world frame
            # Actions are in format: [xyz(3), rotvec(3), gripper(1)]
            local_pose = local_actions[:, :, :6]  # (B, act_horizon, 6)
            gripper_state = local_actions[:, :, 6:]  # (B, act_horizon, act_dim-6)
            
            # Transform each action in the horizon
            world_poses = []
            for h in range(self.act_horizon):
                world_pose_h = transform_flat_pose_vector_th(
                    T=global_frame,
                    v=local_pose[:, h, :],
                    pre_multiply=True
                )
                world_poses.append(world_pose_h)
            
            world_pose = torch.stack(world_poses, dim=1)  # (B, act_horizon, 6)
            world_actions = torch.cat([world_pose, gripper_state], dim=-1)
        
        return {
            "actions": world_actions,
            "base_T": global_frame,
            "local_actions": local_actions,
        }
    
    def compute_loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reference_frame: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the flow matching training loss with frame decoder.
        
        Args:
            obs_seq: Observation sequence, shape (B, obs_horizon, obs_dim)
            action_seq: Action sequence in world frame, shape (B, pred_horizon, act_dim)
            reference_frame: Optional reference frame for training, shape (B, 4, 4)
        
        Returns:
            total_loss: Total loss value
            loss_dict: Dictionary with loss components
        """
        B = obs_seq.shape[0]
        device = obs_seq.device
        obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
        
        total_loss = 0.0
        loss_dict = {}
        
        # Sample aligned noise if sharing
        if self.share_noise:
            action_noise, frame_noise = self._sample_aligned_noise(B, device)
        else:
            action_noise, frame_noise = None, None
        
        # Transform world actions to local frame if using frame decoder
        if self.use_frame_decoder and reference_frame is not None:
            # Use provided reference frame for training
            world_reference_frame = reference_frame
        elif self.use_frame_decoder:
            # Predict reference frame (for consistency, but we'll use GT in loss)
            if frame_noise is not None:
                _, _ = self.frame_decoder(obs_cond, x_init=frame_noise)
            # We'll compute frame loss separately
            world_reference_frame = None
        else:
            world_reference_frame = None
        
        # Transform world actions to local frame
        if world_reference_frame is not None:
            world_pose = action_seq[:, :, :6]  # (B, pred_horizon, 6)
            gripper_state = action_seq[:, :, 6:]  # (B, pred_horizon, act_dim-6)
            
            # Transform each action in the horizon
            local_poses = []
            for h in range(self.pred_horizon):
                # Inverse transform to get local frame
                inv_frame = torch.linalg.inv(world_reference_frame)
                local_pose_h = transform_flat_pose_vector_th(
                    T=inv_frame,
                    v=world_pose[:, h, :],
                    pre_multiply=True
                )
                local_poses.append(local_pose_h)
            
            local_pose = torch.stack(local_poses, dim=1)  # (B, pred_horizon, 6)
            local_action_seq = torch.cat([local_pose, gripper_state], dim=-1)
        else:
            # No frame transformation, use actions as-is
            local_action_seq = action_seq
        
        # Compute action loss using parent's method (but with local actions)
        if action_noise is not None:
            # Use custom noise
            x0 = action_noise
        else:
            x0 = torch.randn(local_action_seq.shape, device=device)
        
        # Sample location along path and get conditional flow
        timestep, xt, ut = self.FM.sample_location_and_conditional_flow(x0, local_action_seq)
        
        # Predict velocity
        vt = self.noise_pred_net(xt, timestep, global_cond=obs_cond)
        
        # Compute action loss
        action_loss = torch.mean((vt - ut) ** 2)
        total_loss += action_loss
        loss_dict["action"] = action_loss.item()
        
        # Compute frame loss if using frame decoder
        if self.use_frame_decoder and reference_frame is not None:
            frame_loss, frame_loss_dict = self.frame_decoder.compute_loss(
                obs_cond=obs_cond,
                gt_poses=reference_frame,
                trans_coeff=self.frame_trans_coeff,
                rot_coeff=self.frame_rot_coeff,
                x_0=frame_noise
            )
            total_loss += self.frame_loss_coeff * frame_loss
            for key, value in frame_loss_dict.items():
                loss_dict[f"frame_{key}"] = value
        
        loss_dict["total"] = total_loss.item()
        
        return total_loss, loss_dict

