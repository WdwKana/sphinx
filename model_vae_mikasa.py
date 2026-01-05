import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from rl_utils.other import device

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ImageEncoder(nn.Module):
    """
    NatureCNN for 128x128x3 RGB images.
    """
    def __init__(self, obs_space):
        super().__init__()
        # obs_space is expected to be a dict with 'image' shape (H, W, C) or just shape tuple
        # But for compatibility with NatureCNN structure:
        
        self.embedding_size = 512
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate output size for 128x128 input
        # 128 -> 31 -> 14 -> 12.  64 * 12 * 12 = 9216
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            out_size = self.cnn(dummy).shape[1]
            
        self.fc = nn.Sequential(
            nn.Linear(out_size, self.embedding_size),
            nn.ReLU()
        )

    def forward(self, obs):
        # Input obs: (B, H, W, C) expected by original code usually?
        # Or (B, C, H, W)?
        # Original code uses obs.image.
        
        if torch.is_tensor(obs):
            x = obs
        else:
            x = obs.image
            
        # Expecting (B, H, W, C) from environment, need (B, C, H, W) for Conv2d
        # Check if channel is last
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
            
        x = x.float() / 255.0
        x = self.cnn(x)
        x = self.fc(x)
        return x

class HistoryEncoder(nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        
        self.image_conv = ImageEncoder(obs_space)
        self.image_embedding_size = self.image_conv.embedding_size
        self.semi_memory_size = 256
        
        # 3-Layer GRU Stack as in original model_vae.py
        self.memory_rnn1 = nn.GRUCell(self.image_embedding_size, self.semi_memory_size)
        self.memory_rnn2 = nn.GRUCell(self.semi_memory_size, self.semi_memory_size)
        self.memory_rnn3 = nn.GRUCell(self.semi_memory_size, self.semi_memory_size)
        
        self.embedding_size = self.semi_memory_size
        
        # Predictor (Context Generator)
        # In original code, prediction is used as context for VAE
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_size + self.image_embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.embedding_size)
        )

    @property
    def memory_size(self):
        # 3 * semi_memory_size because we have 3 stacked GRUs
        return 3 * self.semi_memory_size

    def forward(self, obs, memory):
        x = self.image_conv(obs)
        
        # Split memory into 3 parts
        memory1 = memory[:, :self.semi_memory_size]
        memory2 = memory[:, self.semi_memory_size : 2*self.semi_memory_size]
        memory3 = memory[:, 2*self.semi_memory_size : 3*self.semi_memory_size]

        # Stacked GRU forward pass
        memory1 = self.memory_rnn1(x, memory1)
        memory2 = self.memory_rnn2(F.relu(memory1), memory2)
        memory3 = self.memory_rnn3(F.relu(memory2), memory3)

        # Concatenate memory back
        next_memory = torch.cat([memory1, memory2, memory3], dim=1)
        
        # Generate context (prediction) from (memory1 + memory2) and current obs embedding x
        # This matches original code logic: self.predictor(torch.cat((memory1 + memory2, x), dim=-1))
        # Note: Original code uses memory1 + memory2 (element-wise sum) not concatenation?
        # "torch.cat((memory1 + memory2, x), dim=-1)" implies element-wise sum of m1+m2 then concat with x.
        prediction = self.predictor(torch.cat((memory1 + memory2, x), dim=-1))
        
        # Return context (h_t) and full memory state
        return prediction, next_memory

class BeliefVAEModel(nn.Module):
    def __init__(self, obs_space, state_dim, latent_dim=32):
        super().__init__()
        
        self.state_dim = state_dim
        self.phi_dim = state_dim  # Alias for clarity: VAE state input dim
        self.latent_dim = latent_dim
        
        self.history_model = HistoryEncoder(obs_space)
        self.context_dim = self.history_model.semi_memory_size
        
        # Encoder: q(z | s, h)
        self.vae_encoder = nn.Sequential(
            nn.Linear(state_dim + self.context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim) # mean, softplus(std)
        )
        
        # Decoder: p(s | z, h)
        self.vae_decoder = nn.Sequential(
            nn.Linear(latent_dim + self.context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * state_dim) # mean, softplus(std)
        )
        
        self.apply(init_params)
        self.to(device)

    @property
    def memory_size(self):
        return self.history_model.memory_size

    def forward(self, obs, memory):
        # Standard interface for history update
        return self.history_model(obs, memory)

    def encoder_dist(self, state, context):
        x = torch.cat([state, context], dim=1)
        out = self.vae_encoder(x)
        mean = out[:, :self.latent_dim]
        # Original style: softplus + offset
        std = F.softplus(out[:, self.latent_dim:], beta=1) + 0.1
        return mean, std

    def decoder_dist(self, z, context):
        x = torch.cat([z, context], dim=1)
        out = self.vae_decoder(x)
        mean = out[:, :self.state_dim]
        # Original style
        std = F.softplus(out[:, self.state_dim:], beta=1) + 0.1
        return mean, std

    def sample(self, context):
        # Sample from prior N(0,1) -> Decoder
        # Used for RL inference/visualization
        bs = context.shape[0]
        z = torch.randn(bs, self.latent_dim, device=context.device)
        mean, std = self.decoder_dist(z, context)
        return Normal(mean, std).sample()
