import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torch.distributions.normal import Normal
import torch.distributions.kl as KL
from torch_ac.utils import DictList
from rl_utils import device

class Algo():
    """
    Mikasa-specific VAE Pretraining Algo.
    Adapted from algo_vae_pretrain.py with minimal changes to logic flow.
    """

    def __init__(
        self,
        belief_vae,
        rep_model=None,
        device=None,
        lr_g=0.0003,
        adam_eps=1e-8,
        batch_size_g=128,
        history_recurrence=16,
        beta=0.0001,
        action_dim=None,
        lambda_action=1.0,
    ):
        self.belief_vae = belief_vae
        self.device = device
        self.batch_size_g = batch_size_g
        self.history_recurrence = history_recurrence
        self.beta = beta
        self.rep_model = rep_model
        self.epochs_g = 1 # Added to match loop usage if needed, or controlled outside
        self.lambda_action = lambda_action
        self.action_dim = action_dim

        # Inverse dynamics head (optional if action_dim not provided)
        self.action_id_head = None
        if action_dim is not None:
            self._init_action_modules(action_dim)

        self.vae_optimizer = torch.optim.Adam(self.belief_vae.parameters(), lr_g, eps=adam_eps)
        if self.action_id_head is not None:
            self.vae_optimizer.add_param_group({"params": self.action_id_head.parameters()})

    def _init_action_modules(self, action_dim):
        self.action_dim = action_dim
        # Predict action from concatenated z_t and z_{t+1}
        self.action_id_head = nn.Sequential(
            nn.Linear(self.belief_vae.latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        ).to(self.device)

    def _prepare_action(self, action):
        action = action.to(self.device)
        if action.dim() == 1:
            action = action.unsqueeze(1)
        return action.flatten(start_dim=1)

    def update_g_parameters(self, exps):
        # exps: DictList with obs, state, mask.
        # Dimensions: (T, B, ...)
        
        batch_elbo_loss = 0
        batch_kl = 0
        batch_recon_nll = 0
        batch_action_id = torch.tensor(0.0, device=self.device)
        
        # exps.mask shape: (T, B)
        max_steps, num_episodes = exps.mask.shape[0], exps.mask.shape[1]

        # Initialize memory
        # Mikasa VAE memory size
        memory = torch.zeros((num_episodes, self.belief_vae.history_model.memory_size)).to(self.device)
        prev_z_id = None
        prev_mask = None
        prev_action = None
        total_action_steps = torch.tensor(0.0, device=self.device)

        # Main Loop over time
        for step in range(max_steps):
            sb = exps[step] # Sub-batch at time step

            # Check if any episode is active
            if sb.mask.sum() > 0:
                with torch.no_grad():
                    # Get State Features (Ground Truth State)
                    # Mikasa state is already a vector, just flatten if needed
                    if self.rep_model is not None:
                        rep_encoder_mean, rep_encoder_std = self.rep_model.encode_state(sb.state)
                        state_features = rep_encoder_mean
                    else:
                        state_features = sb.state.to(self.device).flatten(start_dim=1)

                # 1. Forward History / Update Memory
                # Note: Original code detaches memory every 16 steps?
                # "if step % 16 == 0: ... memory.detach() ..."
                # This implements truncated BPTT.
                
                obs_t = sb.obs.to(self.device)
                mask_t = sb.mask.to(self.device).unsqueeze(dim=1)
                
                if step % self.history_recurrence == 0:
                    history_encoding, memory = self.belief_vae(obs_t, memory.detach() * mask_t)
                else:
                    history_encoding, memory = self.belief_vae(obs_t, memory * mask_t)

                # 2. VAE Encoder q(z | s, h)
                encoder_mean, encoder_std = self.belief_vae.encoder_dist(state_features, history_encoding)
                q = Normal(encoder_mean, encoder_std)

                # Learned prior p(z | h)
                prior_mean, prior_std = self.belief_vae.prior_dist(history_encoding)
                p = Normal(prior_mean, prior_std)
                
                # Sample z ~ q(z|s,h) for ELBO, and use z_id for inverse dynamics
                zs = encoder_mean + torch.randn_like(encoder_mean) * encoder_std
                z_id = encoder_mean

                # Optional inverse-dynamics action term via z_{t} -> z_{t+1}
                action_term = None
                action_mask = None
                if prev_z_id is not None and prev_action is not None and hasattr(sb, "action"):
                    action = self._prepare_action(prev_action)
                    if self.action_id_head is None:
                        self._init_action_modules(action.shape[-1])
                        self.vae_optimizer.add_param_group({"params": self.action_id_head.parameters()})
                    action_in = torch.cat([prev_z_id, z_id], dim=1)
                    v_hat = self.action_id_head(action_in)
                    action_term = F.mse_loss(v_hat, action, reduction="none").sum(dim=-1)
                    action_mask = prev_mask * sb.mask.to(self.device)
                
                # 3. VAE Decoder p(s | z, h)
                decoder_mean, decoder_std = self.belief_vae.decoder_dist(zs, history_encoding)
                
                # 4. Calculate ELBO (Monte Carlo Estimate)
                log_px_z = Normal(decoder_mean, decoder_std).log_prob(state_features).sum(dim=-1)
                kl = KL.kl_divergence(q, p).sum(dim=-1)

                # joint objective: recon - beta*KL  (+ action term added separately)
                elbo = log_px_z - self.beta * kl
                
                # Accumulate Loss (Masked)
                # Note: Original code sums negative ELBO
                mask = sb.mask.to(self.device)
                recon_term = -log_px_z
                kl_term = kl

                batch_elbo_loss += -(elbo * mask).sum()
                batch_kl        += (kl_term * mask).sum().detach()
                batch_recon_nll += (recon_term * mask).sum().detach()
                if action_term is not None:
                    batch_elbo_loss += self.lambda_action * (action_term * action_mask).sum()
                    batch_action_id += (action_term * action_mask).sum().detach()
                    total_action_steps += action_mask.sum()

                prev_z_id = z_id
                prev_mask = sb.mask.to(self.device)
                prev_action = sb.action if hasattr(sb, "action") else None

            else:
                # All episodes done
                break

        # Average over total valid steps
        total_valid_steps = exps.mask.sum()
        if total_valid_steps > 0:
            batch_elbo_loss /= total_valid_steps
            batch_kl        /= total_valid_steps
            batch_recon_nll /= total_valid_steps
        if total_action_steps.item() > 0:
            batch_action_id /= total_action_steps

        # Optimization Step
        self.vae_optimizer.zero_grad()
        batch_elbo_loss.backward()

        # Gradient Clipping
        grad_params = list(self.belief_vae.parameters())
        if self.action_id_head is not None:
            grad_params += list(self.action_id_head.parameters())
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in grad_params if p.grad is not None) ** 0.5
        # torch.nn.utils.clip_grad_norm_(self.belief_vae.parameters(), self.gradient_threshold) # Threshold default 200
        
        self.vae_optimizer.step()

        logs = {
            "batch_elbo_loss": batch_elbo_loss.item(),
            "batch_kl": batch_kl.item() if total_valid_steps > 0 else 0.0,
            "batch_recon_nll": batch_recon_nll.item() if total_valid_steps > 0 else 0.0,
            "batch_action_id": batch_action_id.item() if total_valid_steps > 0 else 0.0,
            "grad_norm": grad_norm,
            "total_valid_steps": total_valid_steps.item(),
            "total_action_steps": total_action_steps.item(),
        }
        return logs
