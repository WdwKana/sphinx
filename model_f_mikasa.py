"""
Mikasa-specific Representation Model for Sphinx/Believer.

This model is designed for Mikasa environments where:
- obs (o_t): RGB image (H, W, C) - partial observation
- state (s_t): State vector (D,) - full ground-truth state

Following Believer paper:
- phi(s_t): State encoder (MLP) - captures task-relevant but unobservable information
- psi(o_t): Observation encoder (CNN) - captures observable information
- g(phi(s), psi(o), a): Dynamics model - predicts next state/obs encodings and reward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_params(m):
    """Initialize parameters for linear layers."""
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class RepresentationModelMikasa(nn.Module):
    """
    Representation learning model for Mikasa environments.
    
    Args:
        obs_shape: Shape of RGB observation (H, W, C), e.g., (128, 128, 3)
        state_dim: Dimension of state vector
        action_dim: Dimension of action space
        latent_dim: Dimension of latent representation (default: 16)
        obs_embedding_size: Size of observation embedding after CNN (default: 256)
    """

    def __init__(self, obs_shape, state_dim, action_dim, latent_dim=16, obs_embedding_size=256):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_shape = obs_shape  # (H, W, C)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_embedding_size = obs_embedding_size
        
        print(f"RepresentationModelMikasa initialized:")
        print(f"  obs_shape: {obs_shape}")
        print(f"  state_dim: {state_dim}")
        print(f"  action_dim: {action_dim}")
        print(f"  latent_dim: {latent_dim}")

        # =====================================================================
        # Observation Encoder: CNN for RGB images
        # Input: (B, H, W, C) -> permute to (B, C, H, W)
        # =====================================================================
        in_channels = obs_shape[2]  # C
        
        self.obs_cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, obs_shape[0], obs_shape[1])
            cnn_output_size = self.obs_cnn(dummy_input).shape[1]
        print(f"  CNN output size: {cnn_output_size}")
        
        self.obs_fc = nn.Sequential(
            nn.Linear(cnn_output_size, obs_embedding_size),
            nn.ReLU(),
        )
        
        # psi(o): Observation encoder - outputs mean and std for latent distribution
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)  # mean and log_std
        )

        # =====================================================================
        # State Encoder: MLP for state vectors
        # Input: (B, state_dim)
        # =====================================================================
        # phi(s): State encoder - outputs mean and std for latent distribution
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)  # mean and log_std
        )

        # =====================================================================
        # Dynamics Model: g(z_s, z_o, a) -> (z_s', z_o', r)
        # Input: concatenation of state latent, obs latent, and action
        # =====================================================================
        dynamics_input_dim = 2 * latent_dim + action_dim
        
        self.dynamics_model = nn.Sequential(
            nn.Linear(dynamics_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Prediction heads
        self.next_state_model = nn.Linear(256, latent_dim)
        self.next_obs_model = nn.Linear(256, latent_dim)
        self.reward_model = nn.Linear(256, 1)

        # Initialize parameters
        self.apply(init_params)

    def encode_obs(self, obs):
        """
        Encode observation (RGB image) to latent distribution parameters.
        
        Args:
            obs: RGB image tensor of shape (B, H, W, C) with values in [0, 255] or [0, 1]
            
        Returns:
            encoder_mean: (B, latent_dim)
            encoder_std: (B, latent_dim)
        """
        # Permute from (B, H, W, C) to (B, C, H, W)
        x = obs.float()
        if x.max() > 1.0:
            x = x / 255.0  # Normalize to [0, 1]
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # CNN + FC
        x = self.obs_cnn(x)
        x = self.obs_fc(x)
        
        # Encode to latent distribution
        output = self.obs_encoder(x)
        encoder_mean = output[:, :self.latent_dim]
        encoder_std = F.softplus(output[:, self.latent_dim:], threshold=1) + 1e-5
        
        return encoder_mean, encoder_std

    def encode_state(self, state):
        """
        Encode state vector to latent distribution parameters.
        
        Args:
            state: State tensor of shape (B, state_dim)
            
        Returns:
            encoder_mean: (B, latent_dim)
            encoder_std: (B, latent_dim)
        """
        x = state.float()
        
        # Encode to latent distribution
        output = self.state_encoder(x)
        encoder_mean = output[:, :self.latent_dim]
        encoder_std = F.softplus(output[:, self.latent_dim:], threshold=1) + 1e-5
        
        return encoder_mean, encoder_std

    def predict_next(self, state, obs, action):
        """
        Predict next state/obs latents and reward using dynamics model.
        
        Args:
            state: State tensor of shape (B, state_dim)
            obs: Observation tensor of shape (B, H, W, C)
            action: Action tensor of shape (B, action_dim) or (B,)
            
        Returns:
            next_state_pred: Predicted next state latent (B, latent_dim)
            next_obs_pred: Predicted next obs latent (B, latent_dim)
            reward_pred: Predicted reward (B, 1)
        """
        # Encode state and observation
        encoder_mean_s, encoder_std_s = self.encode_state(state)
        zs = encoder_mean_s + torch.randn_like(encoder_mean_s) * encoder_std_s

        encoder_mean_o, encoder_std_o = self.encode_obs(obs)
        zo = encoder_mean_o + torch.randn_like(encoder_mean_o) * encoder_std_o

        # Ensure action has correct shape
        if len(action.shape) == 1:
            action = action.float().unsqueeze(dim=1)
        else:
            action = action.float()
        
        # Dynamics model
        embedding = self.dynamics_model(torch.cat([zs, zo, action], dim=1))
        
        return (
            self.next_state_model(embedding),
            self.next_obs_model(embedding),
            self.reward_model(embedding)
        )

    def forward(self, state, obs, action):
        """
        Forward pass for training.
        
        Returns all intermediate representations for loss computation.
        """
        # Encode current state and observation
        state_mean, state_std = self.encode_state(state)
        obs_mean, obs_std = self.encode_obs(obs)
        
        # Sample latents
        zs = state_mean + torch.randn_like(state_mean) * state_std
        zo = obs_mean + torch.randn_like(obs_mean) * obs_std
        
        # Ensure action has correct shape
        if len(action.shape) == 1:
            action = action.float().unsqueeze(dim=1)
        else:
            action = action.float()
        
        # Dynamics prediction
        embedding = self.dynamics_model(torch.cat([zs, zo, action], dim=1))
        next_state_pred = self.next_state_model(embedding)
        next_obs_pred = self.next_obs_model(embedding)
        reward_pred = self.reward_model(embedding)
        
        return {
            'state_mean': state_mean,
            'state_std': state_std,
            'obs_mean': obs_mean,
            'obs_std': obs_std,
            'next_state_pred': next_state_pred,
            'next_obs_pred': next_obs_pred,
            'reward_pred': reward_pred,
        }
