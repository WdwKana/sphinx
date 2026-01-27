import torch
import numpy
from torch.distributions.normal import Normal
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
        max_grad_norm=200.0,
    ):
        self.belief_vae = belief_vae
        self.device = device
        self.batch_size_g = batch_size_g
        self.history_recurrence = history_recurrence
        self.beta = beta
        self.rep_model = rep_model
        self.epochs_g = 1 # Added to match loop usage if needed, or controlled outside
        self.max_grad_norm = max_grad_norm
        
        self.vae_optimizer = torch.optim.Adam(self.belief_vae.parameters(), lr_g, eps=adam_eps)

    def update_g_parameters(self, exps):
        # exps: DictList with obs, state, mask.
        # Dimensions: (T, B, ...)
        
        batch_elbo_loss = 0
        batch_kl = 0
        batch_recon_nll = 0
        
        # exps.mask shape: (T, B)
        max_steps, num_episodes = exps.mask.shape[0], exps.mask.shape[1]

        # Initialize memory
        # Mikasa VAE memory size
        memory = torch.zeros((num_episodes, self.belief_vae.history_model.memory_size)).to(self.device)

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

                # Prior p(z) ~ N(0, 1)
                prior_dist = Normal(0, 1)

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
                
                # Sample z
                zs = encoder_mean + torch.randn_like(encoder_mean) * encoder_std
                
                # 3. VAE Decoder p(s | z, h)
                decoder_mean, decoder_std = self.belief_vae.decoder_dist(zs, history_encoding)
                
                # 4. Calculate ELBO (Monte Carlo Estimate)
                # elbo = log p(z) + log p(s|z, h) - log q(z|s, h)
                
                log_pz = prior_dist.log_prob(zs).sum(dim=-1)
                log_px_z = Normal(decoder_mean, decoder_std).log_prob(state_features).sum(dim=-1)
                log_qz_x = Normal(encoder_mean, encoder_std).log_prob(zs).sum(dim=-1)
                
                # ELBO = log_pz + log_px_z - log_qz_x
                elbo = log_pz + log_px_z - log_qz_x
                kl_term = (log_qz_x - log_pz)          # KL(q||p)
                recon_term = (-log_px_z)               # negative log-likelihood
                
                # Accumulate Loss (Masked)
                # Note: Original code sums negative ELBO
                mask = sb.mask.to(self.device)
                batch_elbo_loss += -(elbo * mask).sum()
                batch_kl        += (kl_term * mask).sum()
                batch_recon_nll += (recon_term * mask).sum()

            else:
                # All episodes done
                break

        # Average over total valid steps
        total_valid_steps = exps.mask.sum()
        if total_valid_steps > 0:
            batch_elbo_loss /= total_valid_steps
            batch_kl        /= total_valid_steps
            batch_recon_nll /= total_valid_steps

        # Optimization Step
        self.vae_optimizer.zero_grad()
        batch_elbo_loss.backward()

        # Gradient Clipping
        grad_params = list(self.belief_vae.parameters())
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in grad_params if p.grad is not None) ** 0.5
        torch.nn.utils.clip_grad_norm_(grad_params, self.max_grad_norm)
        
        self.vae_optimizer.step()

        logs = {
            "batch_elbo_loss": batch_elbo_loss.item(),
            "batch_kl": batch_kl.item() if total_valid_steps > 0 else 0.0,
            "batch_recon_nll": batch_recon_nll.item() if total_valid_steps > 0 else 0.0,
            "grad_norm": grad_norm,
            "total_valid_steps": total_valid_steps.item(),
        }
        return logs
