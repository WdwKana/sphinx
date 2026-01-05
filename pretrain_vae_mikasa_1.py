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

# Import Mikasa specific VAE model and Algo
from model_vae_mikasa import BeliefVAEModel
from model_f_mikasa import RepresentationModelMikasa
import algo_vae_pretrain_mikasa as algo_vae_pretrain
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

    ## Parameters for main algorithm
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs for training")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--latent-dim", type=int, default=32,
                        help="number of latent dimensions in VAE")
    parser.add_argument("--beta", type=float, default=0.0001,
                        help="KL loss parameter")

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
    
    # Raw data shapes: (N_episodes, T, ...)
    obss = data["obss"]
    states = data["states"]
    masks = data["masks"]
    
    txt_logger.info(f"Data shapes:")
    txt_logger.info(f"  obss: {obss.shape}")
    txt_logger.info(f"  states: {states.shape}")
    
    # Transpose to (T, N_episodes, ...) for time-loop compatibility in Algo
    obss = obss.transpose(0, 1).to(device)
    states = states.transpose(0, 1).to(device)
    masks = masks.transpose(0, 1).to(device)
    
    obs_shape = tuple(obss.shape[2:]) # (H, W, C)
    state_dim = states.shape[2]
    num_episodes = obss.shape[1]
    rep_model_path = "storage/RememberShapeAndColor3x2-v0_representation_mikasa_seed1/final_model.pt"
    
    if not os.path.exists(rep_model_path):
        raise FileNotFoundError(f"Representation model not found at {rep_model_path}")

    txt_logger.info(f"Loading Rep Model from {rep_model_path}")
    rep_checkpoint = torch.load(rep_model_path, map_location=device)
    
    
    rep_latent_dim = rep_checkpoint.get("args", {}).get("latent_dim", 16) 
    
    rep_model = RepresentationModelMikasa(
        obs_shape=obs_shape,
        state_dim=state_dim,
        action_dim=data["actions"].shape[2] if len(data["actions"].shape) > 2 else 1,
        latent_dim=rep_latent_dim 
    ).to(device)
    phi_dim = rep_latent_dim
    
    rep_model.load_state_dict(rep_checkpoint["model_state"])
    rep_model.eval() # (Freezed)
    for param in rep_model.parameters():
        param.requires_grad = False
    
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
        beta=args.beta
    )
    
    if "vae_optimizer_state" in status:
        algo.vae_optimizer.load_state_dict(status["vae_optimizer_state"])

    # Train Loop
    epochs = status["epochs"]
    update = status["update"]
    start_time = time.time()

    while epochs < args.epochs:
        # Shuffle episode indices
        indices = torch.randperm(num_episodes)
        epoch_losses = []
        epoch_grad_norms = []
        
        # Iterate over batches of EPISODES
        for start_idx in range(0, num_episodes, args.batch_size):
            batch_indices = indices[start_idx : start_idx + args.batch_size]
            
            # Create sub-batch (DictList wrapper)
            # Shapes are (T, B, ...)
            batch_exps = DictList()
            batch_exps.obs = obss[:, batch_indices]
            batch_exps.state = states[:, batch_indices]
            batch_exps.mask = masks[:, batch_indices]
            
            logs = algo.update_g_parameters(batch_exps)
            
            epoch_losses.append(logs["batch_elbo_loss"])
            epoch_grad_norms.append(logs["grad_norm"])
            update += 1
            
            if update % args.log_interval == 0:
                header = ["batch_elbo_loss", "grad_norm"]
                data = [logs["batch_elbo_loss"], logs["grad_norm"]]
                txt_logger.info(f"Epoch {epochs} | Update {update} | Loss: {data[0]:.4f} | Grad: {data[1]:.4f}")
                
                if status["epochs"] == 0 and update == 1:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()
                
                tb_writer.add_scalar("train/elbo_loss", data[0], update)
                tb_writer.add_scalar("train/grad_norm", data[1], update)

        # Save status
        if args.save_interval > 0 and epochs % args.save_interval == 0:
            status = {"epochs": epochs,
                      "update": update,
                      "vae_model_state": vae_model.state_dict(),
                      "vae_optimizer_state": algo.vae_optimizer.state_dict(),
                      }
            torch.save(status, status_path)
            txt_logger.info("Status saved")

        print(f"Epoch {epochs} complete. Avg Loss: {np.mean(epoch_losses):.4f}")
        epochs += 1

    # Save final model
    final_path = os.path.join(model_dir, "final_vae_model.pt")
    torch.save({
        "model_state": vae_model.state_dict(),
        "args": vars(args),
        "obs_shape": obs_shape,
        "state_dim": state_dim,
        "latent_dim": args.latent_dim
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
