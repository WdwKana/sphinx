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

    def __init__(self, env, device=None, adam_eps=1e-8, preprocess_obss=None, lr_g=0.0003, epochs_g=16, rep_model=None,
                 latent_dim=8, latent_dim_f=16, beta=0.0001, gradient_threshold=200, tb_writer=None, use_cnn=False):

        self.env = env
        self.device = device
        self.epochs_g = epochs_g
        self.latent_dim = latent_dim
        self.gradient_threshold = gradient_threshold
        self.rep_model = rep_model
        
        # Mikasa VAE model passed as belief_vae? Or initialized here?
        # In pretrain_vae_mikasa.py we pass env, rep_model etc.
        # But we need to use the passed belief_vae if possible, or init it.
        # Original code inits belief_vae inside __init__.
        # But for Mikasa we might want to pass it or init it with specific params.
        # Let's assume pretrain_vae_mikasa.py passes the model via a hack or we adapt __init__ signature.
        # WAIT: pretrain_vae_mikasa.py calls Algo(env, device, ...). It DOES NOT pass belief_vae instance directly in original code style.
        # BUT I modified pretrain_vae_mikasa.py to pass specific args.
        
        # Let's check how I modified pretrain_vae_mikasa.py:
        # algo = algo_vae_pretrain.Algo(
        #    belief_vae=vae_model, ...
        # )
        # So I changed the signature in my previous response.
        # Let's update this file's __init__ to accept belief_vae directly to be cleaner.
        pass

    # Re-defining __init__ to match the usage in pretrain_vae_mikasa.py
    def __init__(self, belief_vae, rep_model=None, device=None, lr_g=0.0003, adam_eps=1e-8, batch_size_g=128, history_recurrence=16, beta=0.0001, action_dim=None, lambda_action=1.0):
        self.belief_vae = belief_vae
        self.device = device
        self.batch_size_g = batch_size_g
        self.history_recurrence = history_recurrence
        self.beta = beta
        self.rep_model = rep_model
        self.epochs_g = 1 # Added to match loop usage if needed, or controlled outside
        self.lambda_action = lambda_action
        self.action_dim = action_dim
        # Flow-matching style action head: predicts velocity v given xt, t, h, z
        # Input dim: action_dim + 1 (t) + context_dim + latent_dim
        if action_dim is not None:
            self.action_flow_head = nn.Sequential(
                nn.Linear(action_dim + 1 + self.belief_vae.context_dim + self.belief_vae.latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
            ).to(self.device)
        else:
            self.action_flow_head = None
        self.vae_optimizer = torch.optim.Adam(self.belief_vae.parameters(), lr_g, eps=adam_eps)
        if self.action_flow_head is not None:
            self.vae_optimizer.add_param_group({"params": self.action_flow_head.parameters()})

    def update_g_parameters(self, exps):
        # exps: DictList with obs, state, mask.
        # Dimensions: (T, B, ...)
        
        history_encodings = []
        batch_elbo_loss = 0
        batch_kl = 0
        batch_recon_nll = 0
        batch_action_fm = 0
        
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

                # 1. Forward History / Update Memory
                # Note: Original code detaches memory every 16 steps?
                # "if step % 16 == 0: ... memory.detach() ..."
                # This implements truncated BPTT.
                
                obs_t = sb.obs.to(self.device)
                mask_t = sb.mask.to(self.device).unsqueeze(dim=1)
                
                if step % 16 == 0:
                    history_encoding, memory = self.belief_vae(obs_t, memory.detach() * mask_t)
                else:
                    history_encoding, memory = self.belief_vae(obs_t, memory * mask_t)

                # 2. VAE Encoder q(z | s, h)
                encoder_mean, encoder_std = self.belief_vae.encoder_dist(state_features, history_encoding)
                q = Normal(encoder_mean, encoder_std)

                # Learned prior p(z | h)
                prior_mean, prior_std = self.belief_vae.prior_dist(history_encoding)
                p = Normal(prior_mean, prior_std)
                
                # Sample z ~ q(z|s,h)
                zs = encoder_mean + torch.randn_like(encoder_mean) * encoder_std
                
                # 3. VAE Decoder p(s | z, h)
                decoder_mean, decoder_std = self.belief_vae.decoder_dist(zs, history_encoding)
                
                # 4. Calculate ELBO (Monte Carlo Estimate)
                log_px_z = Normal(decoder_mean, decoder_std).log_prob(state_features).sum(dim=-1)
                kl = KL.kl_divergence(q, p).sum(dim=-1)
                
                # Optional action term via flow-matching: predict velocity for noisy action xt
                action_term = 0.0
                if hasattr(sb, "action"):
                    if self.action_flow_head is None:
                        self.action_dim = sb.action.shape[-1]
                        self.action_flow_head = nn.Sequential(
                            nn.Linear(self.action_dim + 1 + self.belief_vae.context_dim + self.belief_vae.latent_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, self.action_dim),
                        ).to(self.device)
                        self.vae_optimizer.add_param_group({"params": self.action_flow_head.parameters()})
                    action = sb.action.to(self.device).flatten(start_dim=1)
                    eps = torch.randn_like(action)
                    t_noise = torch.rand(action.shape[0], 1, device=self.device)
                    xt = t_noise * action + (1 - t_noise) * eps
                    v_target = action - eps  # flow matching target
                    
                    # =====================================================================
                    # [实验 2: Probe(z-only)] 验证 z 是否天然包含动作信息
                    # - 屏蔽 h (用全零替代)，强迫 action head 只能依赖 z
                    # - detach z，阻止 action 梯度回传到 belief_vae
                    # 如果 ActionFM 还能下降，说明 z 确实携带动作信息 (Idea 成立!)
                    # 如果 ActionFM 降不下去，说明动作信息主要不在 z
                    # =====================================================================
                    # >>> 实验开关：取消下面 3 行注释启用 Probe(z-only)，并注释掉原版 <<<
                    #h_for_action = torch.zeros_like(history_encoding)  # 屏蔽 h，防止作弊
                    #z_for_action = zs.detach()                         # 阻断梯度回传
                    #act_in = torch.cat([xt, t_noise, h_for_action, z_for_action], dim=1)
                    
                    # =====================================================================
                    # [原版代码备份] 恢复时：注释掉上面 3 行，取消下面 1 行注释
                    # =====================================================================
                    act_in = torch.cat([xt, t_noise, history_encoding, zs], dim=1)
                    
                    v_hat = self.action_flow_head(act_in)
                    action_term = F.mse_loss(v_hat, v_target, reduction="none").sum(dim=-1)
                
                # joint objective: recon - beta*KL - lambda_action * FM_loss
                elbo = log_px_z - self.beta * kl - self.lambda_action * action_term
                
                # Accumulate Loss (Masked)
                # Note: Original code sums negative ELBO
                mask = sb.mask.to(self.device)
                recon_term = -log_px_z
                kl_term = kl
                action_term_to_log = action_term if torch.is_tensor(action_term) else torch.zeros_like(log_px_z)

                batch_elbo_loss += -(elbo * mask).sum()
                batch_kl        += (kl_term * mask).sum().detach()
                batch_recon_nll += (recon_term * mask).sum().detach()
                batch_action_fm += (action_term_to_log * mask).sum().detach()

            else:
                # All episodes done
                break

        # Average over total valid steps
        total_valid_steps = exps.mask.sum()
        if total_valid_steps > 0:
            batch_elbo_loss /= total_valid_steps
            batch_kl        /= total_valid_steps
            batch_recon_nll /= total_valid_steps
            batch_action_fm /= total_valid_steps

        # Optimization Step
        self.vae_optimizer.zero_grad()
        batch_elbo_loss.backward()

        # Gradient Clipping
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.belief_vae.parameters() if p.grad is not None) ** 0.5
        # torch.nn.utils.clip_grad_norm_(self.belief_vae.parameters(), self.gradient_threshold) # Threshold default 200
        
        self.vae_optimizer.step()

        logs = {
            "batch_elbo_loss": batch_elbo_loss.item(),
            "batch_kl": batch_kl.item() if total_valid_steps > 0 else 0.0,
            "batch_recon_nll": batch_recon_nll.item() if total_valid_steps > 0 else 0.0,
            "batch_action_fm": batch_action_fm.item() if total_valid_steps > 0 else 0.0,
            "grad_norm": grad_norm,
            "total_valid_steps": total_valid_steps.item(),
        }
        return logs
