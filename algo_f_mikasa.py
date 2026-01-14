"""
Mikasa-specific Representation Learning Algorithm.

Changes from algo_f.py:
1. Removed dependency on rl_utils (line 5)
2. Removed env parameter (not needed for offline training)
3. Changed data access from DictList (sb.state, sb.obs) to dict (sb['state'], sb['obs'])
4. Model's encode_state/encode_obs now take raw tensors instead of DictList with .image attribute
"""

import numpy
import torch
from torch.distributions.normal import Normal


class Algo():
    """Representation Learning Algorithm for Mikasa environments."""

    def __init__(self, exps, rep_model, device=None, adam_eps=1e-8, batch_size=256,
                 lr=0.001, beta=0.0001, dynamics_loss_s_coef=0.1, dynamics_loss_o_coef=0.1, reward_loss_coef=0.1, tb_writer=None):

        # Removed: self.env = env (not needed for offline training)
        self.exps = exps
        self.device = device
        self.batch_size = batch_size
        self.max_grad_norm = 0.5
        self.beta = beta
        self.dynamics_loss_s_coef = dynamics_loss_s_coef
        self.dynamics_loss_o_coef = dynamics_loss_o_coef
        self.reward_loss_coef = reward_loss_coef
        self.rep_model = rep_model

        self.optimizer = torch.optim.Adam(self.rep_model.parameters(), lr, eps=adam_eps)
        self.batch_num = 0
        # Changed: access dict key instead of attribute
        self.num_frames = self.exps['next_mask'].shape[0]

        self.tb_writer = tb_writer

    def update_f_parameters(self):

        log_losses = []
        log_state_dynamics_losses = []
        log_obs_dynamics_losses = []
        log_reward_losses = []
        log_kl_losses = []
        log_grad_norms = []
        
        #log_one_rewards = []

        for inds in self._get_batches_starting_indexes():
            # Initialize batch values
            batch_loss = 0
            batch_state_dynamics_loss = 0
            batch_obs_dynamics_loss = 0
            batch_reward_loss = 0
            batch_kl_loss = 0
            sb = {}
            '''
            for key, val in self.exps.items():
                batch_val = val[inds]
                # obs and next_obs are stored as uint8, convert to float here
                if key in ('obs', 'next_obs'):
                    batch_val = batch_val.float()  # uint8 -> float32
                sb[key] = batch_val.to(self.device)
            '''
            # Changed: Create sub-batch using dict access instead of DictList slicing
            for key, val in self.exps.items():
                batch_val = val[inds]
                # Keep obs/next_obs as uint8; encode_obs handles float/normalize on GPU.
                if key in ('obs', 'next_obs'):
                    sb[key] = batch_val.to(self.device, non_blocking=True)
                else:
                    sb[key] = batch_val.to(self.device)
            
            # Changed: Pass tensors directly instead of DictList objects
            # Original: self.rep_model.encode_state(sb.state) where sb.state.image was accessed inside
            # Now: self.rep_model.encode_state(sb['state']) where sb['state'] is the tensor directly
            encoder_mean_s, encoder_std_s = self.rep_model.encode_state(sb['state'])
            encoder_mean_o, encoder_std_o = self.rep_model.encode_obs(sb['obs'])

            next_state_preds, next_obs_preds, reward_preds = self.rep_model.predict_next(sb['state'], sb['obs'], sb['action'])
            next_state_targets, _ = self.rep_model.encode_state(sb['next_state'])
            next_obs_targets, _ = self.rep_model.encode_obs(sb['next_obs'])

            state_dynamics_loss = torch.pow(torch.norm(next_state_preds - (next_state_targets.detach() * sb['next_mask'].unsqueeze(dim=1)), p=2, dim=1), 2).mean()
            obs_dynamics_loss = torch.pow(torch.norm(next_obs_preds - (next_obs_targets.detach() * sb['next_mask'].unsqueeze(dim=1)), p=2, dim=1), 2).mean()
            reward_loss = torch.pow(sb['reward'].unsqueeze(dim=1) - reward_preds, 2).mean()

            # This is E_s[KL(p(Z|S)|| q(Z))] = E_S[E_Z[log(p(Z|S)||q(Z))]]
            kl_loss = torch.distributions.kl.kl_divergence(Normal(encoder_mean_s, encoder_std_s),
                                      Normal(torch.zeros_like(encoder_mean_s),
                                             torch.ones_like(encoder_mean_s)))
            
            #log_one_rewards += reward_preds[(sb['reward'] == 1).nonzero()].flatten().tolist()

            loss = self.beta * kl_loss + self.dynamics_loss_s_coef * state_dynamics_loss + self.dynamics_loss_o_coef * obs_dynamics_loss + self.reward_loss_coef * reward_loss
            
            # Update batch values
            batch_loss += loss.mean()
            batch_state_dynamics_loss += state_dynamics_loss.item()
            batch_obs_dynamics_loss += obs_dynamics_loss.item()
            batch_reward_loss += reward_loss.item()
            batch_kl_loss += kl_loss.mean().item()

            self.optimizer.zero_grad()
            batch_loss.backward()
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.rep_model.parameters() if p.grad is not None) ** 0.5
            #torch.nn.utils.clip_grad_norm_(self.rep_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Update log values
            log_losses.append(batch_loss.item())
            log_state_dynamics_losses.append(batch_state_dynamics_loss)
            log_obs_dynamics_losses.append(batch_obs_dynamics_loss)
            log_reward_losses.append(batch_reward_loss)
            log_kl_losses.append(batch_kl_loss)
            log_grad_norms.append(grad_norm)
        

        torch.set_printoptions(sci_mode=False)
        print("Actual rewards:", [round(x.item(), 3) for x in list(sb['reward'][:10])])
        print("Predicted rewards:", [round(x.item(), 3) for x in list(reward_preds[:10])])

        logs = {
            "grad_norm": numpy.mean(log_grad_norms),
            "state_dynamics_loss": numpy.mean(log_state_dynamics_losses),
            "obs_dynamics_loss": numpy.mean(log_obs_dynamics_losses),
            "reward_loss": numpy.mean(log_reward_losses),
            "kl_loss": numpy.mean(log_kl_losses)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames)
        indexes = numpy.random.permutation(indexes)

        self.batch_num += 1

        num_indexes = self.batch_size
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
