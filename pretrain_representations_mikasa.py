"""
Mikasa-specific Representation Learning Training Script.

Changes from pretrain_representations.py:
1. Import model_f_mikasa instead of model_f (line 6)
2. Import algo_f_mikasa instead of algo_f (line 7)
3. Removed dependency on rl_utils.make_env (line 81-83) - not needed for Mikasa
4. Changed data format: exps is now a dict instead of DictList (line 100-109)
5. Model initialization uses obs_shape, state_dim, action_dim from data (line 86)
6. Algo initialization without env parameter (line 112)
"""

import argparse
import os
import sys
import datetime
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

# Mikasa-specific imports
from model_f_mikasa import RepresentationModelMikasa
import algo_f_mikasa


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--data-path", required=True,
                        help="Path of collected data (.pt file)")
    parser.add_argument("--save-dir", default="storage",
                        help="Directory to save models (default: storage)")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="If set, randomly sample this many episodes (uniform, uses --seed); episode internal order is preserved.",
    )

    ## Parameters for main algorithm
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--epochs", type=int, default=20000,
                        help="number of epochs for training z")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="batch size (default: 1024)")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--latent-dim", type=int, default=16,
                        help="Latent Dimension of representation learning model distribution parameters")
    parser.add_argument("--dynamics-loss-s-coef", type=float, default=0.1,
                        help="dynamics loss parameter")
    parser.add_argument("--dynamics-loss-o-coef", type=float, default=0.1,
                        help="dynamics loss parameter")
    parser.add_argument("--reward-loss-coef", type=float, default=0.1,
                        help="reward loss parameter")    
    parser.add_argument("--beta", type=float, default=0.0001,
                        help="KL loss parameter")

    # ------------------------------------------------------------------
    # State preprocessing (to avoid sentinel outliers like 1000 in z dims)
    # We keep the raw dataset unchanged and apply this only in-memory.
    # Use a replacement > max normal value (e.g., 1 or 10) to preserve the
    # "hidden" indicator without exploding scales.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--state-sentinel",
        type=float,
        default=1000.0,
        help="Sentinel value in state vectors used to encode hidden objects (default: 1000.0).",
    )
    parser.add_argument(
        "--state-sentinel-replace",
        type=float,
        default=None,
        help="If set, replace any state entries equal to --state-sentinel with this value (e.g., 1.0 or 10.0).",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device()
    print(f"Device: {device}")

    # Set run dir
    # Changed: Use custom save_dir instead of rl_utils.get_model_dir
    default_model_name = f"{args.env}_representation_mikasa_seed{args.seed}"
    model_name = args.model or default_model_name
    model_dir = os.path.join(args.save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory: {model_dir}")

    # Log command and all script arguments
    print("{}\n".format(" ".join(sys.argv)))
    print("{}\n".format(args))

    # Load training status
    # Changed: Use simple file-based status instead of rl_utils.get_status
    status_path = os.path.join(model_dir, "status.pt")
    if os.path.exists(status_path):
        status = torch.load(status_path)
        print("Training status loaded from checkpoint\n")
    else:
        status = {"epochs": 0, "update": 0}
        print("Starting fresh training\n")

    # Changed: Removed env loading (rl_utils.make_env_with_env_name)
    # For Mikasa, we get obs_space and action_space from the data file directly

    # Load data
    print(f"Loading data from {args.data_path}...")
    data = torch.load(args.data_path)

    # Episode subsetting (robot RL style): random uniform subset of episodes, reproducible via --seed.
    # Keeps per-episode time order intact; only reduces how many episodes are used.
    if args.num_episodes is not None:
        total_eps = int(data["masks"].shape[0])
        k = int(args.num_episodes)
        if k <= 0:
            raise ValueError("--num-episodes must be > 0")
        k = min(k, total_eps)
        g = torch.Generator()
        g.manual_seed(args.seed)
        ep_ids = torch.randperm(total_eps, generator=g)[:k]
        ep_ids, _ = torch.sort(ep_ids)  # keep natural episode order after random pick

        def _slice_episode_dim(tensor):
            return tensor[ep_ids] if torch.is_tensor(tensor) and tensor.shape[0] == total_eps else tensor

        data = {k_: _slice_episode_dim(v_) for k_, v_ in data.items()}
        print(f"[data] using episodes: {len(ep_ids)}/{total_eps} (seeded random subset)")

    # Extract dimensions from data
    obss = data["obss"]      # (N, T, H, W, C) uint8
    states = data["states"]  # (N, T, D) float
    actions = data["actions"]  # (N, T, action_dim) float
    
    obs_shape = tuple(obss.shape[2:])  # (H, W, C)
    state_dim = states.shape[2]
    action_dim = actions.shape[2] if len(actions.shape) > 2 else 1
    
    print(f"  obss shape: {obss.shape}, obs_shape: {obs_shape}")
    print(f"  states shape: {states.shape}, state_dim: {state_dim}")
    print(f"  actions shape: {actions.shape}, action_dim: {action_dim}")

    # Changed: Load RepresentationModelMikasa instead of RepresentationModel
    # Original: rep_model = RepresentationModel(obs_space=obs_space, action_space=env.action_space, use_cnn='EscapeRoom' in args.env, latent_dim=args.latent_dim)
    rep_model = RepresentationModelMikasa(
        obs_shape=obs_shape,
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim
    ).to(device)
    print("Model loaded\n")
    print("{}\n".format(rep_model))
    if "model_state" in status:
        rep_model.load_state_dict(status["model_state"])

    # Process data: find valid transitions
    eps_len = data["masks"].shape[1]
    indices = (data["masks"] == 1).nonzero(as_tuple=True)
    next_indices = (indices[0], indices[1]+1)
    next_indices_clamped = (indices[0], torch.clamp(indices[1]+1, max=eps_len-1))

    # Changed: Use dict instead of DictList
    # Original:
    #   exps = DictList()
    #   exps.obs = DictList({'image': data["obss"][indices].to(device)})
    #   exps.state = DictList({'image': data["states"][indices].to(device)})
    # Now: Direct tensor access (model_f_mikasa expects raw tensors, not DictList with .image)
    #
    # MEMORY OPTIMIZATION: Keep obs as uint8 on CPU, only convert to float when batching
    # This reduces memory from ~120GB to ~15GB
    exps = {}
    exps['obs'] = data["obss"][indices]                   # (N_valid, H, W, C) - KEEP uint8!
    exps['state'] = data["states"][indices].float()       # (N_valid, D) - float is fine, small
    exps['next_obs'] = data["obss"][next_indices_clamped] # KEEP uint8!
    exps['next_state'] = data["states"][next_indices_clamped].float()
    exps['next_mask'] = F.pad(input=data["masks"], pad=(0, 1))[next_indices].float()
    exps['action'] = data["actions"][indices].float()
    exps['reward'] = data["rewards"][indices].float()

    # Apply sentinel replacement (in-memory only) to stabilize training.
    if args.state_sentinel_replace is not None:
        sentinel = float(args.state_sentinel)
        replace = float(args.state_sentinel_replace)
        shell_game_envs = {"ShellGameTouch-v0", "ShellGamePush-v0", "ShellGamePick-v0"}
        if args.env in shell_game_envs:
            mask_state = exps["state"] >= sentinel
            mask_next_state = exps["next_state"] >= sentinel
            n_s = int(mask_state.sum().item())
            n_ns = int(mask_next_state.sum().item())
            total = n_s + n_ns
            if total > 0:
                print(
                    f"[state_preprocess] shifting state sentinel >= {sentinel} by -990 "
                    f"(state: {n_s}, next_state: {n_ns}, total: {total})"
                )
            exps["state"] = torch.where(mask_state, exps["state"] - 990.0, exps["state"])
            exps["next_state"] = torch.where(mask_next_state, exps["next_state"] - 990.0, exps["next_state"])
            # Original replacement logic (kept for reference):
            # n_s = int((exps["state"] == sentinel).sum().item())
            # n_ns = int((exps["next_state"] == sentinel).sum().item())
            # total = n_s + n_ns
            # if total > 0:
            #     print(
            #         f"[state_preprocess] replacing state sentinel {sentinel} -> {replace} "
            #         f"(state: {n_s}, next_state: {n_ns}, total: {total})"
            #     )
            # exps["state"] = torch.where(
            #     exps["state"] == sentinel,
            #     torch.tensor(replace, dtype=exps["state"].dtype),
            #     exps["state"],
            # )
            # exps["next_state"] = torch.where(
            #     exps["next_state"] == sentinel,
            #     torch.tensor(replace, dtype=exps["next_state"].dtype),
            #     exps["next_state"],
            # )
        else:
            # Count before (for logging)
            n_s = int((exps["state"] == sentinel).sum().item())
            n_ns = int((exps["next_state"] == sentinel).sum().item())
            total = n_s + n_ns
            if total > 0:
                print(
                    f"[state_preprocess] replacing state sentinel {sentinel} -> {replace} "
                    f"(state: {n_s}, next_state: {n_ns}, total: {total})"
                )
            exps["state"] = torch.where(
                exps["state"] == sentinel,
                torch.tensor(replace, dtype=exps["state"].dtype),
                exps["state"],
            )
            exps["next_state"] = torch.where(
                exps["next_state"] == sentinel,
                torch.tensor(replace, dtype=exps["next_state"].dtype),
                exps["next_state"],
            )
    
    # Free original data to save memory
    del data
    import gc
    gc.collect()

    print(f"Total valid transitions: {len(exps['obs'])}")

    # Changed: Load algo_f_mikasa.Algo instead of algo_f.Algo
    # Original: algo = algo_f.Algo(env, exps, rep_model, device, ...)
    # Now: algo_f_mikasa.Algo(exps, rep_model, device, ...) - removed env parameter
    algo = algo_f_mikasa.Algo(
        exps, rep_model, device, args.optim_eps,
        batch_size=args.batch_size, lr=args.lr, tb_writer=None,
        beta=args.beta, dynamics_loss_s_coef=args.dynamics_loss_s_coef,
        dynamics_loss_o_coef=args.dynamics_loss_o_coef, reward_loss_coef=args.reward_loss_coef
    )

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])

    # Open log file
    log_file = open(os.path.join(model_dir, "log.csv"), "a")
    if status["epochs"] == 0:
        log_file.write("epoch,state_dynamics_loss,obs_dynamics_loss,reward_loss,kl_loss,grad_norm\n")

    # Train model
    epochs = status["epochs"]
    update = status["update"]
    start_time = time.time()

    while epochs < args.epochs:
        # Update model parameters
        logs = algo.update_f_parameters()

        # Save status
        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"epochs": epochs,
                      "update": update,
                      "model_state": algo.rep_model.state_dict(),
                      "optimizer_state": algo.optimizer.state_dict(),
                      }
            # Changed: Use torch.save directly instead of rl_utils.save_status
            torch.save(status, status_path)
            print("Status saved")

        # Print logs
        if update % args.log_interval == 0:
            elapsed = time.time() - start_time
            header = ["state_dynamics_loss", "obs_dynamics_loss", "reward_loss", "kl_loss", "grad_norm"]
            log_data = [logs["state_dynamics_loss"], logs["obs_dynamics_loss"], logs["reward_loss"], logs["kl_loss"], logs["grad_norm"]]
            print("Epoch {:5d} | s-loss: {:.4f}, o-loss: {:.4f}, r-loss: {:.4f}, kl-loss: {:.4f}, ∇: {:.4f} | time: {}".format(
                epochs, *log_data, datetime.timedelta(seconds=int(elapsed))))

            log_file.write(f"{epochs},{logs['state_dynamics_loss']},{logs['obs_dynamics_loss']},"
                          f"{logs['reward_loss']},{logs['kl_loss']},{logs['grad_norm']}\n")
            log_file.flush()

        print(epochs, "epochs")
        epochs += 1
        update += 1

    # Save final model
    final_path = os.path.join(model_dir, "final_model.pt")
    torch.save({
        "model_state": rep_model.state_dict(),
        "args": vars(args),
        "obs_shape": obs_shape,
        "state_dim": state_dim,
        "action_dim": action_dim,
        # Explicitly store state preprocessing so downstream VAE training can match it
        "state_sentinel": args.state_sentinel,
        "state_sentinel_replace": args.state_sentinel_replace,
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    log_file.close()
