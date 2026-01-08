import argparse
import time
import datetime
import tensorboardX
import sys
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch_ac.utils import DictList

# Import Mikasa specific VAE model and Algo (CVAE variant)
from model_cvae_mikasa import BeliefVAEModel
from model_f_mikasa import RepresentationModelMikasa
import algo_cvae_pretrain_mikasa as algo_vae_pretrain
import rl_utils # Keep dependency to reuse directory utils

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--env", required=True,
                        help="name of the environment (REQUIRED for naming)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_vae_{TIME})")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--data-path", required=True,
                        help="Path of collected data (.pt file)")
    parser.add_argument("--save-dir", default="storage",
                        help="Directory to save models")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="If set, randomly sample this many episodes (uniform, uses --seed); episode internal order is preserved.",
    )
    parser.add_argument(
        "--rep-model-path",
        default=None,
        help=(
            "Path to pretrained representation model checkpoint (.pt). "
            "If not set, defaults to "
            "storage/{ENV}_representation_mikasa_seed{seed}/final_model.pt"
        ),
    )

    ## Parameters for main algorithm
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training iterations (1 iteration = 1 gradient update on a random episode batch; similar to pretrain_vae.py epochs_g)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--latent-dim", type=int, default=32,
                        help="number of latent dimensions in VAE")
    parser.add_argument("--beta", type=float, default=0.0001,
                        help="KL loss parameter")
    parser.add_argument(
        "--lambda-action",
        dest="lambda_action",
        type=float,
        default=1.0,
        help="Weight for action flow-matching loss in joint ELBO",
    )

    # ------------------------------------------------------------------
    # State preprocessing (avoid sentinel outliers, keep dataset unchanged)
    # If not explicitly set, we will try to inherit this from the rep-model
    # checkpoint (recommended to avoid train/inference mismatch).
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
        help="If set, replace any state entries equal to --state-sentinel with this value (e.g., 1.0 or 10.0). "
             "If not set, and the rep-model checkpoint contains state_sentinel_replace, we will reuse that.",
    )

    args = parser.parse_args()

    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_vae_mikasa_seed{args.seed}"
    model_name = args.model or default_model_name
    model_dir = os.path.join(args.save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Tensorboard writer
    tb_writer = tensorboardX.SummaryWriter(model_dir)
    
    # Log command and args
    # Reuse rl_utils logger if possible or just print
    txt_logger = rl_utils.get_txt_logger(model_dir)
    csv_file, csv_logger = rl_utils.get_csv_logger(model_dir)
    
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    txt_logger.info(f"Device: {device}\n")

    # Load training status
    status_path = os.path.join(model_dir, "status.pt")
    if os.path.exists(status_path):
        status = torch.load(status_path)
        txt_logger.info("Training status loaded\n")
    else:
        status = {"epochs": 0, "update": 0}

    # Load Data
    txt_logger.info(f"Loading data from {args.data_path}...")
    data = torch.load(args.data_path)

    # Episode subsetting: random uniform subset; keeps per-episode time order intact.
    if args.num_episodes is not None:
        total_eps = int(data["masks"].shape[0])
        k = int(args.num_episodes)
        if k <= 0:
            raise ValueError("--num-episodes must be > 0")
        k = min(k, total_eps)
        g = torch.Generator()
        g.manual_seed(args.seed)
        ep_ids = torch.randperm(total_eps, generator=g)[:k]
        ep_ids, _ = torch.sort(ep_ids)

        def _slice_episode_dim(tensor):
            return tensor[ep_ids] if torch.is_tensor(tensor) and tensor.shape[0] == total_eps else tensor

        data = {k_: _slice_episode_dim(v_) for k_, v_ in data.items()}
        txt_logger.info(f"[data] using episodes: {len(ep_ids)}/{total_eps} (seeded random subset)")
    
    # Raw data shapes: (N_episodes, T, ...)
    obss = data["obss"]
    states = data["states"]
    actions = data["actions"]
    masks = data["masks"]
    
    txt_logger.info(f"Data shapes:")
    txt_logger.info(f"  obss: {obss.shape}")
    txt_logger.info(f"  states: {states.shape}")
    
    # Transpose to (T, N_episodes, ...) for time-loop compatibility in Algo
    obss = obss.transpose(0, 1).to(device)
    states = states.transpose(0, 1).to(device)
    actions = actions.transpose(0, 1).to(device)
    masks = masks.transpose(0, 1).to(device)
    
    obs_shape = tuple(obss.shape[2:]) # (H, W, C)
    state_dim = states.shape[2]       # raw state dimension
    action_dim = actions.shape[2] if len(actions.shape) > 2 else 1
    num_episodes = obss.shape[1]
    default_rep_model_path = os.path.join(
        "storage",
        f"{args.env}_representation_mikasa_seed{args.seed}",
        "final_model.pt",
    )
    rep_model_path = args.rep_model_path or default_rep_model_path
    
    if not os.path.exists(rep_model_path):
        raise FileNotFoundError(f"Representation model not found at {rep_model_path}")

    txt_logger.info(f"Loading Rep Model from {rep_model_path}")
    rep_checkpoint = torch.load(rep_model_path, map_location=device)

    # Inherit preprocessing from rep checkpoint unless user overrides
    inherited_replace = None
    inherited_sentinel = None
    if isinstance(rep_checkpoint, dict):
        if "state_sentinel_replace" in rep_checkpoint:
            inherited_replace = rep_checkpoint.get("state_sentinel_replace")
        elif isinstance(rep_checkpoint.get("args"), dict) and "state_sentinel_replace" in rep_checkpoint["args"]:
            inherited_replace = rep_checkpoint["args"].get("state_sentinel_replace")
        if "state_sentinel" in rep_checkpoint:
            inherited_sentinel = rep_checkpoint.get("state_sentinel")
        elif isinstance(rep_checkpoint.get("args"), dict) and "state_sentinel" in rep_checkpoint["args"]:
            inherited_sentinel = rep_checkpoint["args"].get("state_sentinel")

    if args.state_sentinel_replace is None and inherited_replace is not None:
        args.state_sentinel_replace = inherited_replace
    if inherited_sentinel is not None:
        args.state_sentinel = float(inherited_sentinel)
    
    
    rep_latent_dim = rep_checkpoint.get("args", {}).get("latent_dim", 16) 
    
    rep_model = RepresentationModelMikasa(
        obs_shape=obs_shape,
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=rep_latent_dim 
    ).to(device)
    phi_dim = rep_latent_dim

    # Safety check when resuming: ensure phi_dim matches checkpointed metadata
    if "phi_dim" in status and status["phi_dim"] != phi_dim:
        raise ValueError(
            f"Loaded status has phi_dim={status['phi_dim']} but current phi_dim={phi_dim}. "
            "Use the same --rep-model-path (or delete status.pt to restart)."
        )
    
    rep_model.load_state_dict(rep_checkpoint["model_state"])
    rep_model.eval() # (Freezed)
    for param in rep_model.parameters():
        param.requires_grad = False

    # Apply sentinel replacement to raw states in-memory (must match rep training)
    if args.state_sentinel_replace is not None:
        sentinel = float(args.state_sentinel)
        replace = float(args.state_sentinel_replace)
        n_before = int((states == sentinel).sum().item())
        if n_before > 0:
            txt_logger.info(
                f"[state_preprocess] replacing state sentinel {sentinel} -> {replace} "
                f"(count={n_before})"
            )
        states = torch.where(states == sentinel, torch.tensor(replace, device=states.device, dtype=states.dtype), states)
    
    # Init Model
    vae_model = BeliefVAEModel(
        obs_space=obs_shape,
        state_dim=phi_dim,
        latent_dim=args.latent_dim
    ).to(device)
    
    txt_logger.info(f"Model loaded:\n{vae_model}\n")
    
    if "vae_model_state" in status:
        vae_model.load_state_dict(status["vae_model_state"])

    # Init Algo
    algo = algo_vae_pretrain.Algo(
        belief_vae=vae_model,
        rep_model=rep_model,
        device=device,
        lr_g=args.lr,
        adam_eps=args.optim_eps,
        batch_size_g=args.batch_size,
        beta=args.beta,
        action_dim=action_dim,
        lambda_action=args.lambda_action,
    )
    
    # Resume action head (if present) then optimizer
    if "action_head_state" in status and getattr(algo, "action_flow_head", None) is not None:
        missing, unexpected = algo.action_flow_head.load_state_dict(status["action_head_state"], strict=False)
        txt_logger.info(f"Loaded action_flow_head state (missing={len(missing)}, unexpected={len(unexpected)})")
    if "vae_optimizer_state" in status:
        try:
            algo.vae_optimizer.load_state_dict(status["vae_optimizer_state"])
        except ValueError as e:
            txt_logger.info(
                f"Optimizer state mismatch (likely new param groups); reinitializing optimizer. Err={e}"
            )

    # Train Loop
    epochs = status.get("epochs", 0)
    update = status.get("update", 0)
    # Backward-compat: older versions used `epochs` as full dataset sweeps while `update`
    # tracked gradient steps. This script treats 1 epoch == 1 gradient update, so we
    # align `epochs` to the update counter when resuming from older checkpoints.
    if epochs != update:
        txt_logger.info(
            f"Note: loaded status has epochs={epochs}, update={update}. "
            "This script uses 1 epoch = 1 update; aligning epochs to update.\n"
        )
        epochs = update
    start_time = time.time()

    while epochs < args.epochs:
        # Match pretrain_vae.py semantics:
        # one "epoch" == one gradient update on a randomly sampled batch of EPISODES
        epoch_idx = epochs
        batch_size = min(args.batch_size, num_episodes)
        batch_indices = torch.randperm(num_episodes)[:batch_size]

        # Create sub-batch (DictList wrapper)
        # Shapes are (T, B, ...)
        batch_exps = DictList()
        batch_exps.obs = obss[:, batch_indices]
        batch_exps.state = states[:, batch_indices]
        batch_exps.action = actions[:, batch_indices]
        batch_exps.mask = masks[:, batch_indices]

        logs = algo.update_g_parameters(batch_exps)

        update += 1
        epochs += 1

        if update % args.log_interval == 0:
            header = [
                "batch_elbo_loss",
                "batch_recon_nll",
                "batch_kl",
                "batch_action_fm",
                "grad_norm",
                "total_valid_steps",
            ]
            data = [
                logs.get("batch_elbo_loss", 0.0),
                logs.get("batch_recon_nll", 0.0),
                logs.get("batch_kl", 0.0),
                logs.get("batch_action_fm", 0.0),
                logs.get("grad_norm", 0.0),
                logs.get("total_valid_steps", 0),
            ]
            txt_logger.info(
                f"Epoch {epoch_idx} | Update {update} | "
                f"ELBO: {data[0]:.4f} | ReconNLL: {data[1]:.4f} | KL: {data[2]:.4f} | "
                f"ActionFM: {data[3]:.4f} | Grad: {data[4]:.4f} | Valid steps: {data[5]}"
            )

            if status.get("epochs", 0) == 0 and update == 1:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            tb_writer.add_scalar("train/elbo_loss", data[0], update)
            tb_writer.add_scalar("train/recon_nll", data[1], update)
            tb_writer.add_scalar("train/kl", data[2], update)
            tb_writer.add_scalar("train/action_fm", data[3], update)
            tb_writer.add_scalar("train/grad_norm", data[4], update)

        # Save status (based on update count, consistent with CLI help text)
        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {
                "epochs": epochs,
                "update": update,
                "vae_model_state": vae_model.state_dict(),
                "vae_optimizer_state": algo.vae_optimizer.state_dict(),
                # action head (flow) state if exists
                "action_head_state": algo.action_flow_head.state_dict()
                if getattr(algo, "action_flow_head", None) is not None
                else None,
                # Metadata for correct resume/debug
                "phi_dim": phi_dim,
                "raw_state_dim": state_dim,
                "rep_model_path": rep_model_path,
                "rep_latent_dim": rep_latent_dim,
            }
            torch.save(status, status_path)
            txt_logger.info("Status saved")

        print(f"Epoch {epoch_idx} complete. Loss: {logs['batch_elbo_loss']:.4f}")

    # Save final model
    final_path = os.path.join(model_dir, "final_vae_model.pt")
    torch.save({
        "model_state": vae_model.state_dict(),
        "args": vars(args),
        "obs_shape": obs_shape,
        # VAE uses phi_dim as its state embedding size
        "state_dim": phi_dim,
        "phi_dim": phi_dim,
        "raw_state_dim": state_dim,
        "latent_dim": args.latent_dim,
        "rep_model_path": rep_model_path,
        "rep_latent_dim": rep_latent_dim,
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
