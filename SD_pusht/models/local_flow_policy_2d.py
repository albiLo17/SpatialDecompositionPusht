"""Local Flow Matching model with 2D position decoder for spatial decomposition.

This is a simplified 2D version for tasks like PushT where we predict a 2D reference
position instead of a 3D reference frame, and use simple translation instead of SE(3) transformation.
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .conditional_unet import ConditionalUnet1D, SinusoidalPosEmb
from .conditional_flow_matcher import ConditionalFlowMatcher
from .flow_matching import FlowMatching

# Import normalization functions
try:
    from SD_pusht.utils.normalization import normalize_data, unnormalize_data
except ImportError:
    # Fallback if not available (define locally)
    def normalize_data(data, stats):
        """Normalize data to [-1, 1] range."""
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
        ndata = ndata * 2 - 1
        return ndata
    
    def unnormalize_data(ndata, stats):
        """Unnormalize data from [-1, 1] range."""
        ndata = (ndata + 1) / 2
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        return data


class FiLMReEncoder2D(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) reencoder for 2D position conditioning.
    
    Adapts the encoded observation features based on the reference position using
    multiplicative (scale) and additive (bias) conditioning.
    Similar to FiLMReEncoder in spatialdecomposition but for 2D positions.
    """
    
    def __init__(
        self,
        feature_dim: int,
        position_dim: int = 2,
        hidden_dim: int = 32,
        predict_scale: bool = True,
    ):
        """
        Args:
            feature_dim: Dimension of the feature vector to be conditioned
            position_dim: Dimension of the position (2 for 2D)
            hidden_dim: Hidden dimension for the position encoder MLP
            predict_scale: If True, predict both scale and bias (full FiLM).
                          If False, only predict bias (additive conditioning only).
        """
        super().__init__()
        
        self.predict_scale = predict_scale
        self.cond_channels = feature_dim
        if predict_scale:
            self.cond_channels *= 2  # Need 2x channels for scale and bias
        
        self.position_encoder = nn.Sequential(
            nn.Linear(position_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, self.cond_channels),
        )
    
    def forward(self, features: torch.Tensor, conditioning_position: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning to features based on position.
        
        Args:
            features: Feature vector to condition, shape (B, feature_dim)
            conditioning_position: Reference position, shape (B, position_dim)
        
        Returns:
            Conditioned features, shape (B, feature_dim)
        """
        cond_params = self.position_encoder(conditioning_position)  # (B, cond_channels)
        
        if self.predict_scale:
            cond_params = cond_params.reshape(
                cond_params.shape[:-1] + (2, self.cond_channels // 2)
            )  # (B, 2, feature_dim)
            scale = cond_params[:, 0, :]  # (B, feature_dim)
            bias = cond_params[:, 1, :]   # (B, feature_dim)
            
            assert scale.shape == features.shape, \
                f"Scale shape {scale.shape} != features shape {features.shape}"
        else:
            scale = 1.0
            bias = cond_params
        
        assert bias.shape == features.shape, \
            f"Bias shape {bias.shape} != features shape {features.shape}"
        
        out = scale * features + bias
        return out


class ObservationEncoder(nn.Module):
    """
    Simple MLP encoder for observations.
    
    Takes flattened observations and encodes them to a fixed dimension (128).
    This encoder is shared between the position decoder and the action prediction network.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        activation: str = "mish",
    ):
        """
        Args:
            input_dim: Input dimension (obs_horizon * obs_dim)
            output_dim: Output dimension (default: 128)
            activation: Activation function ("mish", "relu", "gelu", "tanh")
        """
        super().__init__()
        
        # Choose activation
        if activation == "mish":
            act_fn = nn.Mish()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Two small layers: input_dim -> intermediate -> output_dim
        # Use intermediate dimension as mean of input and output
        intermediate_dim = (input_dim + output_dim) // 2
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            act_fn,
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, output_dim),
            act_fn,
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs_flat: Flattened observations, shape (B, input_dim)
        
        Returns:
            Encoded observations, shape (B, output_dim)
        """
        return self.encoder(obs_flat)


class PositionMLP(nn.Module):
    """
    MLP backbone for position prediction with timestep and observation conditioning.
    
    This is a simpler alternative to the U-Net architecture for debugging and comparison.
    """
    
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        hidden_dims: list = None,
        activation: str = "mish",
    ):
        """
        Args:
            input_dim: Input dimension (2 for 2D position)
            global_cond_dim: Dimension of global conditioning (observation)
            diffusion_step_embed_dim: Dimension of timestep embedding
            hidden_dims: List of hidden layer dimensions (default: [256, 512, 256])
            activation: Activation function ("mish", "relu", "gelu", "tanh")
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]
        
        # Timestep embedding (same as U-Net)
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        
        # Choose activation
        if activation == "mish":
            act_fn = nn.Mish()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Input: [x, timestep_emb, global_cond]
        # x: (B, T, input_dim) -> flattened to (B, T*input_dim) = (B, input_dim) for T=1
        # timestep_emb: (B, dsed)
        # global_cond: (B, global_cond_dim)
        input_features = input_dim + dsed + global_cond_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn)
            layers.append(nn.LayerNorm(hidden_dim))  # Layer normalization for stability
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sample: Input position, shape (B, T, input_dim) where T=1
            timestep: Timestep, shape (B,)
            global_cond: Global conditioning (observations), shape (B, global_cond_dim)
        
        Returns:
            Output velocity, shape (B, T, input_dim)
        """
        B = sample.shape[0]
        device = sample.device
        
        # Flatten temporal dimension: (B, T, input_dim) -> (B, T*input_dim)
        # For T=1, this is just (B, input_dim)
        x_flat = sample.flatten(start_dim=1)  # (B, input_dim)
        
        # Encode timestep
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float, device=device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(device)
        timestep = timestep.expand(B)
        
        timestep_emb = self.diffusion_step_encoder(timestep)  # (B, dsed)
        
        # Concatenate: [x, timestep_emb, global_cond]
        if global_cond is not None:
            x_combined = torch.cat([x_flat, timestep_emb, global_cond], dim=-1)
        else:
            x_combined = torch.cat([x_flat, timestep_emb], dim=-1)
        
        # Pass through MLP
        output = self.mlp(x_combined)  # (B, input_dim)
        
        # Reshape to match expected output format: (B, T, input_dim)
        output = output.unsqueeze(1)  # (B, 1, input_dim)
        
        return output


class DirectPositionMLP(nn.Module):
    """
    Direct MLP for position prediction without flow matching.
    
    This is a simple regression MLP that directly predicts positions from observations,
    without using flow matching. Useful for comparison with flow-based approaches.
    """
    
    def __init__(
        self,
        obs_cond_dim: int,
        hidden_dims: list = None,
        activation: str = "mish",
    ):
        """
        Args:
            obs_cond_dim: Dimension of observation conditioning
            hidden_dims: List of hidden layer dimensions (default: [256, 512, 256])
            activation: Activation function ("mish", "relu", "gelu", "tanh")
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]
        
        # Choose activation
        if activation == "mish":
            act_fn = nn.Mish()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Input: observation conditioning only
        # Output: 2D position
        input_features = obs_cond_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn)
            layers.append(nn.LayerNorm(hidden_dim))  # Layer normalization for stability
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 2))  # 2D position (x, y)
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, obs_cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - direct position prediction.
        
        Args:
            obs_cond: Observation conditioning, shape (B, obs_cond_dim)
        
        Returns:
            position: Predicted 2D position, shape (B, 2)
        """
        return self.mlp(obs_cond)


class Position2DFlowDecoder(nn.Module):
    """
    Flow matching decoder for predicting a 2D reference position.
    
    This is a simplified version for 2D tasks where we only need to predict
    a translation (x, y) instead of a full pose.
    """
    
    def __init__(
        self,
        obs_cond_dim: int,
        sigma: float = 0.0,
        fm_timesteps: int = 8,
        diffusion_step_embed_dim: int = 256,
        down_dims: list = None,
        n_groups: int = 4,
        backbone: str = "unet",
        mlp_hidden_dims: list = None,
        mlp_activation: str = "mish",
        use_flow_matching: bool = True,
        num_particles: int = 1,
        particles_aggregation: str = "median",  # "median" or "knn"
    ):
        """
        Args:
            obs_cond_dim: Dimension of observation conditioning
            sigma: Noise scale for flow matching paths (only used if use_flow_matching=True)
            fm_timesteps: Number of timesteps for flow matching inference (only used if use_flow_matching=True)
            diffusion_step_embed_dim: Dimension of timestep embedding (only used if use_flow_matching=True)
            down_dims: Channel dimensions for U-Net downsampling (only used if backbone="unet" and use_flow_matching=True)
            n_groups: Number of groups for GroupNorm (only used if backbone="unet" and use_flow_matching=True)
            backbone: Backbone architecture ("unet", "mlp", or "mlp_direct")
            mlp_hidden_dims: Hidden layer dimensions for MLP
            mlp_activation: Activation function for MLP ("mish", "relu", "gelu", "tanh")
            use_flow_matching: If False, use direct regression instead of flow matching (only works with "mlp" backbone)
            num_particles: Number of particles to sample for position prediction (only used if use_flow_matching=True).
                If > 1, samples multiple positions and aggregates them. Default: 1 (single sample).
            particles_aggregation: Method to aggregate particles when num_particles > 1.
                Options: "median" (element-wise median) or "knn" (KNN-based density estimation).
                Default: "median".
        """
        super().__init__()
        
        self.POSITION_DIM = 2  # 2D position (x, y)
        self.fm_timesteps = fm_timesteps
        self.backbone = backbone.lower()
        self.use_flow_matching = use_flow_matching
        self.num_particles = num_particles
        self.particles_aggregation = particles_aggregation
        
        if particles_aggregation not in ["median", "knn"]:
            raise ValueError(
                f"Unknown particles_aggregation method: {particles_aggregation}. "
                f"Must be 'median' or 'knn'."
            )
        
        # Handle direct MLP (no flow matching)
        if self.backbone == "mlp_direct" or (self.backbone == "mlp" and not use_flow_matching):
            self.use_flow_matching = False
            self.position_net = DirectPositionMLP(
                obs_cond_dim=obs_cond_dim,
                hidden_dims=mlp_hidden_dims,
                activation=mlp_activation,
            )
            # No flow matcher needed for direct regression
            self.FM = None
        elif self.backbone == "unet":
            if down_dims is None:
                # For single timestep prediction (T=1), we avoid downsampling/upsampling
                # to prevent temporal dimension mismatches. With T=1, upsampling doubles
                # the temporal dimension, causing concatenation errors in skip connections.
                # Using a single level ensures the last level uses Identity() for down/up,
                # preserving T=1.
                down_dims = [256]
            elif len(down_dims) > 1:
                # Warn if multiple levels are used with T=1, as this can cause issues
                import warnings
                warnings.warn(
                    f"Position2DFlowDecoder uses T=1 (single timestep). "
                    f"Using {len(down_dims)} downsampling levels may cause temporal dimension "
                    f"mismatches. Consider using a single level (e.g., [256]).",
                    UserWarning
                )
            
            # Flow matching network for position prediction using U-Net
            # Note: For single timestep (T=1), we need to be careful with downsampling.
            # The U-Net architecture uses the last level's Identity() for down/up sampling,
            # so with a single level, there's no actual down/up sampling, preserving T=1.
            # With multiple levels, upsampling doubles T, causing concatenation errors.
            self.position_net = ConditionalUnet1D(
                input_dim=self.POSITION_DIM,
                global_cond_dim=obs_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=3,  # Use kernel_size=3 for better behavior
                n_groups=n_groups,
            )
            # Conditional Flow Matcher (required for U-Net with flow matching)
            self.FM = ConditionalFlowMatcher(sigma=sigma)
        elif self.backbone == "mlp":
            # Flow matching network for position prediction using MLP
            self.position_net = PositionMLP(
                input_dim=self.POSITION_DIM,
                global_cond_dim=obs_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                hidden_dims=mlp_hidden_dims,
                activation=mlp_activation,
            )
            # Conditional Flow Matcher
            self.FM = ConditionalFlowMatcher(sigma=sigma)
        else:
            raise ValueError(
                f"Unknown backbone: {backbone}. Must be 'unet', 'mlp', or 'mlp_direct'. "
                f"Or use 'mlp' with use_flow_matching=False for direct regression."
            )
    
    def forward(
        self,
        obs_cond: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
        k_clusters: Optional[int] = None,
        return_particles: bool = False
    ) -> torch.Tensor:
        """
        Predict 2D position from observation conditioning.
        
        Args:
            obs_cond: Observation conditioning, shape (B, obs_cond_dim)
            x_init: Optional initial noise (only used if use_flow_matching=True)
                If provided and num_particles > 1, this will be used as the first particle.
            k_clusters: Optional number of nearest neighbors for KNN aggregation (only used if particles_aggregation="knn").
                If None, uses default k = max(1, P // 2) where P is num_particles.
                This allows test-time ablation of the number of clusters.
            return_particles: If True, returns a dict with 'position', 'all_particles', and 'selected_index'.
                Otherwise, just returns the position tensor. Default: False (backward compatible).
        
        Returns:
            If return_particles=False:
                position: 2D position, shape (B, 2)
                    If num_particles > 1, returns the median across particles or KNN-selected particle.
            If return_particles=True:
                dict with:
                    - 'position': 2D position, shape (B, 2)
                    - 'all_particles': All particle positions, shape (B, num_particles, 2) or None
                    - 'selected_index': Index of selected particle per batch, shape (B,) or None
        """
        if not self.use_flow_matching:
            # Direct regression: MLP(obs) -> position
            position = self.position_net(obs_cond)
            if return_particles:
                return {
                    'position': position,
                    'all_particles': None,
                    'selected_index': None
                }
            return position
        
        # Flow matching inference
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        # If using particles, sample multiple positions and take median
        if self.num_particles > 1:
            all_positions = []
            
            with torch.no_grad():
                # Sample particles
                for p in range(self.num_particles):
                    # Use provided x_init for first particle if available, otherwise sample
                    if p == 0 and x_init is not None:
                        noisy_position = x_init
                        assert noisy_position.shape == (B, 1, self.POSITION_DIM), \
                            f"Expected x_init shape ({B}, 1, {self.POSITION_DIM}), got {noisy_position.shape}"
                    else:
                        noisy_position = torch.randn(
                            (B, 1, self.POSITION_DIM), device=device
                        )
                    
                    # Integrate the flow over timesteps for this particle
                    dt = 1.0 / self.fm_timesteps
                    denoised_position = noisy_position
                    
                    for i in range(self.fm_timesteps):
                        t = i / self.fm_timesteps
                        timestep = torch.tensor([t], device=device).repeat(B,)
                        
                        vt = self.position_net(
                            denoised_position, timestep, global_cond=obs_cond
                        )
                        denoised_position = (vt * dt + denoised_position)
                    
                    # Extract position for this particle
                    particle_position = denoised_position.squeeze(1)  # (B, POSITION_DIM)
                    all_positions.append(particle_position)
            
            # Stack all particle positions: (num_particles, B, POSITION_DIM)
            all_positions = torch.stack(all_positions, dim=0)  # (num_particles, B, POSITION_DIM)
            
            # Transpose to (B, num_particles, POSITION_DIM) for easier batch processing
            all_positions_bt = all_positions.transpose(0, 1)  # (B, num_particles, POSITION_DIM)
            
            # Aggregate particles based on method
            if self.particles_aggregation == "median":
                # Element-wise median across particles: (B, POSITION_DIM)
                position = torch.median(all_positions, dim=0)[0]
                if return_particles:
                    return {
                        'position': position,
                        'all_particles': all_positions_bt,  # (B, num_particles, POSITION_DIM)
                        'selected_index': None  # Median doesn't have a single selected index
                    }
            elif self.particles_aggregation == "knn":
                # KNN-based density estimation
                B, P, _ = all_positions_bt.shape
                
                # Calculate pairwise euclidean distances: (B, num_particles, num_particles)
                euclidean_distances = torch.norm(
                    (all_positions_bt[:, :, None, :] - all_positions_bt[:, None, :, :]),
                    dim=-1,
                    p=2,
                )
                
                # Exclude self-distances (set diagonal to inf for proper KNN density estimation)
                mask = torch.eye(P, device=euclidean_distances.device, dtype=torch.bool).unsqueeze(0)  # (1, P, P)
                euclidean_distances = euclidean_distances.masked_fill(mask, float('inf'))  # (B, P, P)
                
                # For each particle, find k nearest neighbors
                # Then pick the particle with smallest mean distance to its neighbors
                # Allow test-time ablation of k_clusters
                if k_clusters is not None:
                    k = max(1, min(k_clusters, P - 1))  # Use specified k, but ensure it's valid
                else:
                    k = max(1, P // 2)  # Default: use P//2 neighbors (backward compatible)
                particle_idx = (
                    euclidean_distances.topk(k=k, dim=-1, sorted=False, largest=False)[0]  # (B, num_particles, k)
                    .mean(dim=-1)  # Mean distance to k nearest neighbors: (B, num_particles)
                    .argmin(dim=-1)  # Pick particle with smallest mean: (B,)
                )
                
                # Select the chosen particle for each batch element
                position = all_positions_bt[range(all_positions_bt.shape[0]), particle_idx]  # (B, POSITION_DIM)
                
                if return_particles:
                    return {
                        'position': position,
                        'all_particles': all_positions_bt,  # (B, num_particles, POSITION_DIM)
                        'selected_index': particle_idx  # (B,)
                    }
            else:
                raise ValueError(
                    f"Unknown particles_aggregation method: {self.particles_aggregation}"
                )
            
            if return_particles:
                return {
                    'position': position,
                    'all_particles': all_positions_bt,
                    'selected_index': None
                }
            
            return position
        
        # Single particle inference (original behavior)
        with torch.no_grad():
            # Initialize from noise (single timestep, so horizon=1)
            if x_init is None:
                noisy_position = torch.randn(
                    (B, 1, self.POSITION_DIM), device=device
                )
            else:
                noisy_position = x_init
                assert noisy_position.shape == (B, 1, self.POSITION_DIM), \
                    f"Expected x_init shape ({B}, 1, {self.POSITION_DIM}), got {noisy_position.shape}"
            
            # Integrate the flow over timesteps
            # Use Euler integration with step size dt = 1.0 / fm_timesteps
            # This integrates from t=0 to t≈1.0 (reaches 1.0 - dt)
            dt = 1.0 / self.fm_timesteps
            for i in range(self.fm_timesteps):
                # Timestep for velocity evaluation: t = i * dt
                # This goes from 0 to (fm_timesteps-1)/fm_timesteps
                # To reach t=1.0, we could use t = min(1.0, i / (fm_timesteps - 1)) for last step
                # But standard practice is to use i/fm_timesteps and accept slight under-integration
                t = i / self.fm_timesteps
                timestep = torch.tensor([t], device=device).repeat(B,)
                
                if i == 0:
                    vt = self.position_net(
                        noisy_position, timestep, global_cond=obs_cond
                    )
                    denoised_position = (vt * dt + noisy_position)
                else:
                    vt = self.position_net(
                        denoised_position, timestep, global_cond=obs_cond
                    )
                    denoised_position = (vt * dt + denoised_position)
        
        # Extract position
        position = denoised_position.squeeze(1)  # (B, POSITION_DIM)
        
        if return_particles:
            return {
                'position': position,
                'all_particles': None,
                'selected_index': None
            }
        
        return position
    
    def compute_loss(
        self,
        obs_cond: torch.Tensor,
        gt_positions: torch.Tensor,
        x_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss for position prediction.
        
        If use_flow_matching=True: Uses flow matching loss.
        If use_flow_matching=False: Uses direct MSE loss on predicted positions.
        
        Args:
            obs_cond: Observation conditioning, shape (B, obs_cond_dim)
            gt_positions: Ground truth 2D positions, shape (B, 2)
            x_0: Optional pre-sampled noise (only used if use_flow_matching=True)
        
        Returns:
            total_loss: Total loss value
            loss_dict: Dictionary with individual loss components
        """
        if not self.use_flow_matching:
            # Direct regression: predict position and compute MSE
            pred_positions = self.position_net(obs_cond)  # (B, 2)
            loss = F.mse_loss(pred_positions, gt_positions)
            
            loss_dict = {
                "total": loss.item(),
            }
            
            return loss, loss_dict
        
        # Flow matching loss
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        # Add horizon dimension for consistency with network
        gt_position_flat = gt_positions.unsqueeze(1)  # (B, 2) -> (B, 1, 2)
        
        # Sample noise
        if x_0 is None:
            x0 = torch.randn(gt_position_flat.shape, device=device)
        else:
            x0 = x_0
            assert x0.shape == gt_position_flat.shape, \
                f"Expected x_0 shape {gt_position_flat.shape}, got {x0.shape}"
        
        # Sample location along path and get conditional flow
        timestep, xt, ut = self.FM.sample_location_and_conditional_flow(x0, gt_position_flat)
        
        # Predict velocity
        vt = self.position_net(xt, timestep, global_cond=obs_cond)
        
        # Compute MSE loss
        loss = F.mse_loss(vt, ut)
        
        loss_dict = {
            "total": loss.item(),
        }
        
        return loss, loss_dict


class LocalFlowPolicy2D(FlowMatching):
    """
    Local Flow Matching policy with 2D position decoder for spatial decomposition.
    
    This extends FlowMatching for 2D tasks (like PushT) where we predict actions in a local
    frame (relative to a 2D reference position), then transform them to the world frame
    using simple translation.
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
        # Position decoder parameters
        use_position_decoder: bool = True,
        position_decoder_down_dims: list = None,
        position_decoder_n_groups: int = 4,
        position_decoder_fm_timesteps: int = 8,
        position_decoder_num_particles: int = 1,
        position_decoder_particles_aggregation: str = "median",  # "median" or "knn"
        # Loss coefficients
        position_loss_coeff: float = 1.0,
        # Noise sharing
        share_noise: bool = False,
        shared_noise_base: str = "action",  # "action", "position", or "combinatory"
        # Oracle mode: use ground truth reference position for local policy conditioning during training
        use_gt_reference_for_local_policy: bool = False,
        # Transform observations to local frame (relative to reference position)
        transform_obs_to_local_frame: bool = False,
        # FiLM conditioning: use FiLM (Feature-wise Linear Modulation) for position conditioning
        use_film_conditioning: bool = False,
        film_hidden_dim: int = 32,
        film_predict_scale: bool = True,
        # Ablation: disable reference position conditioning for local policy
        disable_reference_conditioning: bool = False,
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
            use_position_decoder: Whether to use position decoder
            position_decoder_down_dims: Channel dimensions for position decoder U-Net
            position_decoder_n_groups: Number of groups for position decoder GroupNorm
            position_decoder_fm_timesteps: Number of timesteps for position decoder inference
            position_decoder_num_particles: Number of particles to sample for position prediction.
                If > 1, samples multiple positions and aggregates them. Default: 1 (single sample).
            position_decoder_particles_aggregation: Method to aggregate particles when num_particles > 1.
                Options: "median" (element-wise median) or "knn" (KNN-based density estimation).
                Default: "median".
            position_loss_coeff: Overall coefficient for position loss
            share_noise: Whether to share noise between position and action predictions
            shared_noise_base: Base for noise sharing ("action", "position", or "combinatory")
            use_gt_reference_for_local_policy: If True, use ground truth reference position 
                for local policy conditioning during training (oracle mode). This allows 
                training the local policy with privileged information to understand the 
                best possible performance.
            transform_obs_to_local_frame: If True, transform observations to local frame 
                (relative to reference position) before using them. For PushT, this subtracts 
                the reference position from agent_x, agent_y, block_x, block_y (first 4 dims).
            use_film_conditioning: If True, use FiLM (Feature-wise Linear Modulation) to condition
                observation features with reference position. Otherwise, use concatenation (default).
            film_hidden_dim: Hidden dimension for FiLM position encoder MLP
            film_predict_scale: If True, FiLM predicts both scale and bias. If False, only bias.
            disable_reference_conditioning: If True, disable reference position conditioning for the local policy.
                The reference position will still be used for action transformation (to local frame), but won't
                be concatenated or used for FiLM conditioning. This is useful for ablation studies (baseline).
                Default: False (use reference conditioning).
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
        
        self.use_position_decoder = use_position_decoder
        self.share_noise = share_noise
        self.shared_noise_base = shared_noise_base
        self.position_loss_coeff = position_loss_coeff
        self.use_gt_reference_for_local_policy = use_gt_reference_for_local_policy
        self.transform_obs_to_local_frame = transform_obs_to_local_frame
        self.use_film_conditioning = use_film_conditioning
        self.disable_reference_conditioning = disable_reference_conditioning
        
        # Create shared observation encoder
        obs_flat_dim = obs_horizon * obs_dim
        self.obs_encoder_dim = 128
        self.obs_encoder = ObservationEncoder(
            input_dim=obs_flat_dim,
            output_dim=self.obs_encoder_dim,
            activation="mish",
        )
        
        # Create FiLM reencoder if enabled
        if use_film_conditioning:
            self.pose_reencoder = FiLMReEncoder2D(
                feature_dim=self.obs_encoder_dim,
                position_dim=2,  # 2D position (x, y)
                hidden_dim=film_hidden_dim,
                predict_scale=film_predict_scale,
            )
        else:
            self.pose_reencoder = None
        
        if use_position_decoder:
            # Position decoder now uses encoded observations
            self.position_decoder = Position2DFlowDecoder(
                obs_cond_dim=self.obs_encoder_dim,  # Use encoded dimension
                sigma=sigma,
                fm_timesteps=position_decoder_fm_timesteps,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=position_decoder_down_dims,
                n_groups=position_decoder_n_groups,
                num_particles=position_decoder_num_particles,
                particles_aggregation=position_decoder_particles_aggregation,
            )
            # Recreate noise_pred_net
            # If using FiLM, global_cond_dim is just obs_encoder_dim (position is modulated into features)
            # Otherwise, global_cond_dim is obs_encoder_dim + 2 (position is concatenated)
            REF_POSITION_DIM = 2  # 2D reference position (x, y)
            if use_film_conditioning:
                # With FiLM, position is modulated into features, so global_cond_dim is just obs_encoder_dim
                noise_cond_dim = self.obs_encoder_dim
            else:
                # Without FiLM, concatenate position with obs encoding
                noise_cond_dim = self.obs_encoder_dim + REF_POSITION_DIM
            # Apply same default as parent class if down_dims is None
            noise_net_down_dims = down_dims if down_dims is not None else [256, 512, 1024]
            self.noise_pred_net = ConditionalUnet1D(
                input_dim=self.act_dim,
                global_cond_dim=noise_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=noise_net_down_dims,
                n_groups=n_groups,
            )
        else:
            self.position_decoder = None
            # Still need to update noise_pred_net to use encoded observations
            noise_net_down_dims = down_dims if down_dims is not None else [256, 256]
            self.noise_pred_net = ConditionalUnet1D(
                input_dim=self.act_dim,
                global_cond_dim=self.obs_encoder_dim,  # Use encoded obs
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=noise_net_down_dims,
                n_groups=n_groups,
            )
    
    def _sample_aligned_noise(
        self,
        B: int,
        device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sample aligned noise for action and position predictions.
        
        Args:
            B: Batch size
            device: Device to sample on
        
        Returns:
            action_noise: Noise for action prediction, shape (B, pred_horizon, act_dim) or None
            position_noise: Noise for position prediction, shape (B, 1, 2) or None
        """
        if not self.share_noise:
            return None, None
        
        H = self.pred_horizon
        P = 2  # POSITION_DIM
        A = self.act_dim
        
        if self.shared_noise_base == "action":
            # Sample action noise first, then project to position noise
            action_noise = torch.randn((B, H, A), device=device)
            # Aggregate along horizon dimension
            position_noise = action_noise.mean(dim=1, keepdim=True)  # (B, 1, A)
            variance_rescale = np.sqrt(H)
            position_noise = position_noise * variance_rescale
            # Project from action_dim to 2D (take first 2 dims or pad)
            if A >= 2:
                position_noise = position_noise[:, :, :2]
            else:
                padding = torch.zeros((B, 1, 2 - A), device=device)
                position_noise = torch.cat([position_noise, padding], dim=-1)
        
        elif self.shared_noise_base == "position":
            # Sample position noise first, then project to action noise
            position_noise = torch.randn((B, 1, P), device=device)
            # Expand to action noise
            action_noise = position_noise.repeat(1, H, 1)
            variance_rescale = np.sqrt(H)
            action_noise = action_noise / variance_rescale
            # Project from 2D to action_dim
            if A <= 2:
                action_noise = action_noise[:, :, :A]
            else:
                padding = torch.randn((B, H, A - 2), device=device) / variance_rescale
                action_noise = torch.cat([action_noise, padding], dim=-1)
        
        elif self.shared_noise_base == "combinatory":
            # Sample from a higher-dimensional space and project
            full_noise = torch.randn((B, H, P * A), device=device)
            # Simple projection: take mean for position, reshape for action
            position_noise = full_noise.mean(dim=1, keepdim=True) * np.sqrt(H)
            position_noise = position_noise[:, :, :P]
            action_noise = full_noise[:, :, :A]
        else:
            raise ValueError(f"Unknown shared_noise_base: {self.shared_noise_base}")
        
        return action_noise, position_noise
    
    def _transform_obs_to_local_frame(
        self,
        obs_seq: torch.Tensor,
        reference_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform observations to local frame (relative to reference position).
        
        For PushT, observations are [agent_x, agent_y, block_x, block_y, block_angle].
        We subtract the reference position from the first 4 dimensions (positions).
        
        Args:
            obs_seq: Observation sequence in world frame, shape (B, obs_horizon, obs_dim)
            reference_position: Reference position, shape (B, 2)
        
        Returns:
            Transformed observation sequence in local frame, shape (B, obs_horizon, obs_dim)
        """
        local_obs_seq = obs_seq.clone()
        # Subtract reference position from first 4 dimensions (agent_x, agent_y, block_x, block_y)
        # For PushT: obs_dim=5, first 4 are positions
        ref_pos_expanded = reference_position.unsqueeze(1)  # (B, 1, 2)
        # Transform agent position (dims 0, 1) and block position (dims 2, 3)
        local_obs_seq[:, :, :2] = obs_seq[:, :, :2] - ref_pos_expanded  # agent position
        local_obs_seq[:, :, 2:4] = obs_seq[:, :, 2:4] - ref_pos_expanded  # block position
        # block_angle (dim 4) stays the same
        return local_obs_seq
    
    def get_action(
        self,
        obs_seq: torch.Tensor,
        reference_position: Optional[torch.Tensor] = None,
        action_stats: Optional[Dict] = None,
        reference_pos_stats: Optional[Dict] = None,
        k_clusters: Optional[int] = None,
        return_particles: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample actions from the local flow matching model.
        
        Args:
            obs_seq: Observation sequence, shape (B, obs_horizon, obs_dim)
            reference_position: Optional reference 2D position for training, shape (B, 2)
            action_stats: Optional action statistics for normalization
            reference_pos_stats: Optional reference position statistics for normalization
            k_clusters: Optional number of nearest neighbors for KNN aggregation (only used if particles_aggregation="knn").
                If None, uses default k = max(1, P // 2) where P is num_particles. Default: None (backward compatible).
            return_particles: If True, returns particle information in the action_dict. Default: False (backward compatible).
        
        Returns:
            Dictionary with:
                - "actions": World frame actions, shape (B, act_horizon, act_dim)
                - "reference_pos": Predicted reference position, shape (B, 2)
                - "local_actions": Local frame actions, shape (B, act_horizon, act_dim)
                - "particles_info": (optional, if return_particles=True) Dict with 'particles' and 'selected_index'
        """
        B = obs_seq.shape[0]
        device = obs_seq.device
        
        with torch.no_grad():
            # First, get reference position (needed for observation transformation)
            # Use original observations for position decoder
            obs_flat_for_position = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
            obs_encoded_for_position = self.obs_encoder(obs_flat_for_position)  # (B, obs_encoder_dim)
            
            # Predict reference position if using position decoder
            if self.use_position_decoder:
                if self.share_noise:
                    _, position_noise = self._sample_aligned_noise(B, device)
                else:
                    position_noise = None
                
                ref_position_result = self.position_decoder(
                    obs_encoded_for_position, 
                    x_init=position_noise,
                    k_clusters=k_clusters,
                    return_particles=return_particles
                )
                
                if return_particles and isinstance(ref_position_result, dict):
                    ref_position = ref_position_result['position']
                    particles_info = {
                        'particles': ref_position_result.get('all_particles'),
                        'selected_index': ref_position_result.get('selected_index'),
                    }
                else:
                    ref_position = ref_position_result if isinstance(ref_position_result, torch.Tensor) else ref_position_result['position']
                    particles_info = None
            else:
                # Use provided reference position or zero
                if reference_position is not None:
                    ref_position = reference_position
                else:
                    ref_position = torch.zeros((B, 2), device=device)
            
            # Transform observations to local frame if requested
            if self.transform_obs_to_local_frame and ref_position is not None:
                obs_seq_local = self._transform_obs_to_local_frame(obs_seq, ref_position)
            else:
                obs_seq_local = obs_seq
            
            # Flatten and encode observations for conditioning
            obs_flat_local = obs_seq_local.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
            obs_cond = self.obs_encoder(obs_flat_local)  # (B, obs_encoder_dim)
            
            # Apply FiLM conditioning if enabled (skip if reference conditioning is disabled)
            if (self.use_position_decoder and self.use_film_conditioning and 
                ref_position is not None and not self.disable_reference_conditioning):
                obs_cond = self.pose_reencoder(obs_cond, ref_position)  # (B, obs_encoder_dim)
            
            # Sample aligned noise for actions if sharing
            if self.share_noise:
                action_noise, _ = self._sample_aligned_noise(B, device)
            else:
                action_noise = None
            
            # Get local actions using parent's flow matching
            if action_noise is not None:
                noisy_action_seq = action_noise  # (B, pred_horizon, act_dim)
            else:
                noisy_action_seq = torch.randn(
                    (B, self.pred_horizon, self.act_dim), device=device
                )
            
            # Prepare conditioning: use FiLM if enabled, otherwise concatenate
            # Skip reference conditioning if disabled (ablation/baseline)
            if (self.use_position_decoder and ref_position is not None and 
                not self.disable_reference_conditioning):
                if self.use_film_conditioning:
                    # FiLM already applied above, use obs_cond directly
                    action_cond = obs_cond
                else:
                    # Concatenate obs_cond with ref_position for local action prediction
                    action_cond = torch.cat([obs_cond, ref_position], dim=-1)  # (B, obs_encoder_dim + 2)
            else:
                # No reference conditioning (ablation mode or no reference position available)
                action_cond = obs_cond
            
            # Integrate the flow over timesteps
            for i in range(self.fm_timesteps):
                timestep = torch.tensor(
                    [i / self.fm_timesteps], device=device
                ).repeat(B,)
                
                if i == 0:
                    vt = self.noise_pred_net(
                        noisy_action_seq, timestep, global_cond=action_cond
                    )
                    denoised_action_seq = (vt * 1 / self.fm_timesteps + noisy_action_seq)
                else:
                    vt = self.noise_pred_net(
                        denoised_action_seq, timestep, global_cond=action_cond
                    )
                    denoised_action_seq = (vt * 1 / self.fm_timesteps + denoised_action_seq)
            
            # Extract local actions (act_horizon portion)
            start = self.obs_horizon - 1
            end = start + self.act_horizon
            local_actions = denoised_action_seq[:, start:end]  # (B, act_horizon, act_dim) - normalized local actions
            
            # Transform local actions to world frame
            # Steps: unnormalize local actions → transform to world frame → return unnormalized world actions
            # This matches the training logic: unnormalize, transform, then we'll normalize in evaluation if needed
            if action_stats is not None and reference_pos_stats is not None:
                # Unnormalize local actions (from normalized space to world coordinates)
                local_actions_np = local_actions.detach().cpu().numpy()
                local_actions_unnorm = unnormalize_data(local_actions_np, action_stats)
                
                # Unnormalize reference position
                ref_pos_np = ref_position.detach().cpu().numpy()
                ref_pos_unnorm = unnormalize_data(ref_pos_np, reference_pos_stats)
                
                # Transform to world frame: add reference position to first 2 dims
                world_actions_unnorm = local_actions_unnorm.copy()
                world_actions_unnorm[:, :, :2] = local_actions_unnorm[:, :, :2] + ref_pos_unnorm[:, None, :]
                
                # Convert back to tensor (unnormalized world actions)
                world_actions = torch.from_numpy(world_actions_unnorm).float().to(device)
            else:
                # Fallback: simple addition in normalized space (incorrect but won't crash)
                world_actions = local_actions.clone()
                world_actions[:, :, :2] = local_actions[:, :, :2] + ref_position.unsqueeze(1)
        
        result = {
            "actions": world_actions,
            "reference_pos": ref_position,
            "local_actions": local_actions,
        }
        
        # Add particle information if requested
        if return_particles and particles_info is not None:
            result["particles_info"] = particles_info
        
        return result
    
    def compute_loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reference_position: Optional[torch.Tensor] = None,
        action_stats: Optional[Dict] = None,
        reference_pos_stats: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the flow matching training loss with position decoder.
        
        Args:
            obs_seq: Observation sequence, shape (B, obs_horizon, obs_dim)
            action_seq: Action sequence in world frame, shape (B, pred_horizon, act_dim)
            reference_position: Optional reference 2D position for training, shape (B, 2)
        
        Returns:
            total_loss: Total loss value
            loss_dict: Dictionary with loss components
        """
        B = obs_seq.shape[0]
        device = obs_seq.device
        
        total_loss = 0.0
        loss_dict = {}
        
        # Sample aligned noise if sharing
        if self.share_noise:
            action_noise, position_noise = self._sample_aligned_noise(B, device)
        else:
            action_noise, position_noise = None, None
        
        # Determine which reference position to use for local policy conditioning
        # First, get reference position using original (world frame) observations
        obs_flat_for_position = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
        obs_encoded_for_position = self.obs_encoder(obs_flat_for_position)  # (B, obs_encoder_dim)
        
        # Option 1: Use GT reference position (oracle mode) if flag is set and GT is available
        if self.use_gt_reference_for_local_policy and reference_position is not None:
            # Oracle mode: use ground truth reference position for local policy conditioning
            world_ref_position_for_conditioning = reference_position
        # Option 2: Use predicted reference position (normal mode)
        elif self.use_position_decoder and reference_position is not None:
            # Use provided reference position for training (but we'll predict it for conditioning)
            # Predict reference position for conditioning to match inference behavior
            if position_noise is not None:
                world_ref_position_for_conditioning = self.position_decoder(obs_encoded_for_position, x_init=position_noise)
            else:
                world_ref_position_for_conditioning = self.position_decoder(obs_encoded_for_position, x_init=None)
        elif self.use_position_decoder:
            # Predict reference position for conditioning (even if we don't have GT)
            # This ensures the local policy is aware of the predicted frame during training
            if position_noise is not None:
                world_ref_position_for_conditioning = self.position_decoder(obs_encoded_for_position, x_init=position_noise)
            else:
                world_ref_position_for_conditioning = self.position_decoder(obs_encoded_for_position, x_init=None)
        else:
            world_ref_position_for_conditioning = None
        
        # For transforming actions to local frame, always use GT if available
        # We should use GT reference position whenever it's provided, regardless of use_position_decoder
        # This is especially important when use_position_decoder=False but we still want to train on local actions
        if reference_position is not None:
            world_ref_position_for_transform = reference_position
        else:
            world_ref_position_for_transform = None
        
        # Transform observations to local frame if requested
        if self.transform_obs_to_local_frame and world_ref_position_for_conditioning is not None:
            obs_seq_local = self._transform_obs_to_local_frame(obs_seq, world_ref_position_for_conditioning)
        else:
            obs_seq_local = obs_seq
        
        # Flatten and encode observations for conditioning
        obs_flat_local = obs_seq_local.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
        obs_cond = self.obs_encoder(obs_flat_local)  # (B, obs_encoder_dim)
        
        # Apply FiLM conditioning if enabled (before action transformation)
        # Skip if reference conditioning is disabled (ablation/baseline)
        if (self.use_position_decoder and self.use_film_conditioning and 
            world_ref_position_for_conditioning is not None and 
            not self.disable_reference_conditioning):
            obs_cond = self.pose_reencoder(obs_cond, world_ref_position_for_conditioning)  # (B, obs_encoder_dim)
        
        # Transform world actions to local frame
        # Steps: unnormalize world actions → transform to local frame → normalize local actions
        if world_ref_position_for_transform is not None and action_stats is not None and reference_pos_stats is not None:
            # Unnormalize world actions (from normalized space to world coordinates)
            # Convert to numpy for unnormalization (temporary)
            action_seq_np = action_seq.detach().cpu().numpy()
            world_actions_unnorm = unnormalize_data(action_seq_np, action_stats)
            
            # Unnormalize reference position using predefined stats
            ref_pos_np = world_ref_position_for_transform.detach().cpu().numpy()
            ref_pos_unnorm = unnormalize_data(ref_pos_np, reference_pos_stats)
            
            # Transform to local frame: subtract reference position from first 2 dims
            local_actions_unnorm = world_actions_unnorm.copy()
            local_actions_unnorm[:, :, :2] = world_actions_unnorm[:, :, :2] - ref_pos_unnorm[:, None, :]
            
            # Normalize local actions back to [-1, 1] range
            # Use the same normalization stats as world actions (assuming similar scale)
            # Alternatively, we could compute new stats for local actions, but using same stats is simpler
            local_action_seq = torch.from_numpy(
                normalize_data(local_actions_unnorm, action_stats)
            ).float().to(action_seq.device)
        else:
            # Fallback: simple subtraction in normalized space (old behavior, less correct)
            if world_ref_position_for_transform is not None:
                local_action_seq = action_seq.clone()
                local_action_seq[:, :, :2] = action_seq[:, :, :2] - world_ref_position_for_transform.unsqueeze(1)
            else:
                local_action_seq = action_seq
        
        # Compute action loss using parent's method (but with local actions)
        if action_noise is not None:
            x0 = action_noise
        else:
            x0 = torch.randn(local_action_seq.shape, device=device)
        
        # Sample location along path and get conditional flow
        timestep, xt, ut = self.FM.sample_location_and_conditional_flow(x0, local_action_seq)
        
        # Prepare conditioning: use FiLM if enabled (already applied above), otherwise concatenate
        # The local policy should be aware of the reference frame (GT in oracle mode, predicted otherwise)
        # Skip reference conditioning if disabled (ablation/baseline)
        if (self.use_position_decoder and world_ref_position_for_conditioning is not None and 
            not self.disable_reference_conditioning):
            if not self.use_film_conditioning:
                # Concatenate obs_cond with reference position for local action prediction
                action_cond = torch.cat([obs_cond, world_ref_position_for_conditioning], dim=-1)  # (B, obs_encoder_dim + 2)
            else:
                # FiLM already applied above, use obs_cond directly
                action_cond = obs_cond
        else:
            # No reference conditioning (ablation mode or no reference position available)
            action_cond = obs_cond
        
        # Predict velocity
        vt = self.noise_pred_net(xt, timestep, global_cond=action_cond)
        
        # Compute action loss
        action_loss = torch.mean((vt - ut) ** 2)
        total_loss += action_loss
        loss_dict["action"] = action_loss.item()
        
        # Compute position loss if using position decoder
        # Note: We always train the position decoder with GT, even in oracle mode
        # Position decoder uses original (world frame) observations, not transformed ones
        if self.use_position_decoder and reference_position is not None:
            # Use encoded original observations for position decoder (world frame)
            # obs_encoded_for_position was already computed above
            position_loss, position_loss_dict = self.position_decoder.compute_loss(
                obs_cond=obs_encoded_for_position,
                gt_positions=reference_position,
                x_0=position_noise
            )
            total_loss += self.position_loss_coeff * position_loss
            for key, value in position_loss_dict.items():
                loss_dict[f"position_{key}"] = value
        
        loss_dict["total"] = total_loss.item()
        
        return total_loss, loss_dict

