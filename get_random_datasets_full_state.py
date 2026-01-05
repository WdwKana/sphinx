"""
Random policy data collection for Sphinx-style belief/VAE training.

Collects data using random actions and saves in Sphinx-compatible .pt format:
- obss: rgb observations (N_episodes, T, H, W, C) - partial observation
- joints: robot joints info (N_episodes, T, J) - robot proprioception (part of observation)
- states: full state vector (N_episodes, T, D) - ground-truth state (includes joints)
- actions: random actions (N_episodes, T, action_dim)
- rewards: rewards (N_episodes, T)
- masks: valid step masks (N_episodes, T) - 1 for valid, 0 for done

Following Believer paper:
- obss + joints = o_t (Observation)
- states = s_t (Ground Truth State)

Usage:
    python3 mikasa_robo_suite/dataset_collectors/get_random_datasets_full_state.py \
        --env-id=ShellGamePick-v0 \
        --path-to-save-data="data" \
        --num-episodes=1000 \
        --batch-size=128
"""

import os
import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from typing import Optional

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper
from mikasa_robo_suite.memory_envs import *
from mikasa_robo_suite.utils.wrappers import *


def env_info(env_id):
    """
    Get environment-specific wrapper configuration and episode timeout.
    """
    noop_steps = 1
    
    if env_id in ['ShellGamePush-v0', 'ShellGamePick-v0', 'ShellGameTouch-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (ShellGameRenderCupInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        EPISODE_TIMEOUT = 90
    elif env_id in ['InterceptSlow-v0', 'InterceptMedium-v0', 'InterceptFast-v0', 
                    'InterceptGrabSlow-v0', 'InterceptGrabMedium-v0', 'InterceptGrabFast-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        EPISODE_TIMEOUT = 90
    elif env_id in ['RotateLenientPos-v0', 'RotateLenientPosNeg-v0',
                    'RotateStrictPos-v0', 'RotateStrictPosNeg-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (RotateRenderAngleInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        EPISODE_TIMEOUT = 90
    elif env_id in ['CameraShutdownPush-v0', 'CameraShutdownPick-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (CameraShutdownWrapper, {"n_initial_steps": 19}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
        ]
        EPISODE_TIMEOUT = 90
    elif env_id in ['TakeItBack-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        EPISODE_TIMEOUT = 180
    elif env_id in ['RememberColor3-v0', 'RememberColor5-v0', 'RememberColor9-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RememberColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        EPISODE_TIMEOUT = 60
    elif env_id in ['RememberShape3-v0', 'RememberShape5-v0', 'RememberShape9-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RememberShapeInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        EPISODE_TIMEOUT = 60
    elif env_id in ['RememberShapeAndColor3x2-v0', 'RememberShapeAndColor3x3-v0', 'RememberShapeAndColor5x3-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RememberShapeAndColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        EPISODE_TIMEOUT = 60
    elif env_id in ['BunchOfColors3-v0', 'BunchOfColors5-v0', 'BunchOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        EPISODE_TIMEOUT = 120
    elif env_id in ['SeqOfColors3-v0', 'SeqOfColors5-v0', 'SeqOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        EPISODE_TIMEOUT = 120
    elif env_id in ['ChainOfColors3-v0', 'ChainOfColors5-v0', 'ChainOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        EPISODE_TIMEOUT = 120
    else:
        raise ValueError(f"Unknown environment: {env_id}")
    
    wrappers_list.insert(0, (StateOnlyTensorToDictWrapper, {}))

    return wrappers_list, EPISODE_TIMEOUT


def collect_random_data_sphinx_format(
    env_id: str = "ShellGameTouch-v0", 
    path_to_save_data: str = "data",
    num_episodes: int = 1000,
    batch_size: int = 128,
    seed: int = 42
):
    """
    Collect data using random policy and save in Sphinx-compatible .pt format.
    
    Strategy:
    - Use SINGLE environment for perfect alignment.
    - Wrapper: rgb=True, state=True, joints=True.
    - Output 'obss': RGB images.
    - Output 'joints': Proprioception data (part of Observation o).
    - Output 'states': Partial State + Joints (Full Ground Truth s).
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get environment config
    wrappers_list, episode_timeout = env_info(env_id)
    
    # Calculate number of batches needed
    num_batches = (num_episodes + batch_size - 1) // batch_size
    actual_num_episodes = num_batches * batch_size
    
    print(f"Environment: {env_id}")
    print(f"Episode timeout: {episode_timeout}")
    print(f"Collecting {actual_num_episodes} episodes in {num_batches} batches (batch_size={batch_size})")
    
    # =========================================================================
    # Create ONE environment
    # =========================================================================
    
    # Environment kwargs
    # Important: obs_mode="rgb" enables rendering. We rely on wrappers to also get state.
    env_kwargs = dict(
        obs_mode="rgb", 
        control_mode="pd_joint_delta_pos",
        render_mode="all",
        sim_backend="gpu",
        reward_mode="normalized_dense"
    )
    
    print("Creating synchronized environment...")
    env = gym.make(env_id, num_envs=batch_size, **env_kwargs)
    for wrapper_class, wrapper_kwargs in wrappers_list:
        env = wrapper_class(env, **wrapper_kwargs)
    
    # Configure wrapper to return ALL info
    # joints=True -> extracts joints into obs['joints'], leaves obs['state'] as Partial State
    env = FlattenRGBDObservationWrapper(
        env, 
        rgb=True,
        depth=False,
        state=True,    # Include State (Partial if joints=True)
        oracle=True,   # Include oracle_info (Requires env mod)
        joints=True    # Extract joints separately
    )
    
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, batch_size, ignore_terminations=True, record_metrics=True)
    
    # Get action space bounds for random sampling
    action_low = torch.tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    action_dim = env.single_action_space.shape[0]
    
    # =========================================================================
    # Get observation dimensions from first reset
    # =========================================================================
    print("Resetting environment to get observation dimensions...")
    obs, _ = env.reset(seed=seed)
    
    # 1. RGB Shape
    if 'rgb' in obs:
        rgb_shape = (obs['rgb'].shape[1], obs['rgb'].shape[2], 3)  # (H, W, 3)
        print(f"RGB shape: {rgb_shape}")
    else:
        raise ValueError("Environment did not return RGB. Check obs_mode='rgb'")
    
    # 2. Joints Dim
    if 'joints' in obs:
        joints_dim = obs['joints'].shape[-1]
        print(f"Joints dim: {joints_dim}")
    else:
        joints_dim = 0
        print("Warning: No joints info available")
    
    # 3. Partial State Dim
    if 'state' in obs:
        state_dim = obs['state'].shape[-1]  # <--- 修改：直接使用 Partial State Dim 作为最终 State Dim
        print(f"State dim (Partial, no joints): {state_dim}")
    else:
        raise ValueError("Environment did not return State")
    
    # 4. Full State Dim (Target)
    # We NO LONGER concatenate Partial State + Joints.
    # states will only contain the hidden/environment info, not the joints.
    # state_dim = partial_state_dim + joints_dim <--- 删除这行
    
    # =========================================================================
    # Pre-allocate tensors for all episodes
    # =========================================================================
    all_obss = torch.zeros((actual_num_episodes, episode_timeout, *rgb_shape), dtype=torch.uint8)
    all_joints = torch.zeros((actual_num_episodes, episode_timeout, joints_dim), dtype=torch.float32)
    all_states = torch.zeros((actual_num_episodes, episode_timeout, state_dim), dtype=torch.float32)
    all_actions = torch.zeros((actual_num_episodes, episode_timeout, action_dim), dtype=torch.float32)
    all_rewards = torch.zeros((actual_num_episodes, episode_timeout), dtype=torch.float32)
    all_masks = torch.zeros((actual_num_episodes, episode_timeout), dtype=torch.float32)
    
    # =========================================================================
    # Collect data batch by batch
    # =========================================================================
    print(f"\nStarting data collection...")
    for batch_idx in tqdm(range(num_batches), desc="Collecting batches"):
        episode_start_idx = batch_idx * batch_size
        
        current_seed = seed + batch_idx
        obs, _ = env.reset(seed=current_seed)
        
        active_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        
        for t in range(episode_timeout):
            # 1. Store RGB
            if 'rgb' in obs:
                all_obss[episode_start_idx:episode_start_idx + batch_size, t] = obs['rgb'][..., :3].cpu().to(torch.uint8)
            
            # 2. Store Joints
            current_joints = obs['joints'].cpu().float()
            all_joints[episode_start_idx:episode_start_idx + batch_size, t] = current_joints
            
            # 3. Store State (ONLY Partial State, NO Joints)
            current_partial_state = obs['state'].cpu().float()
            # current_full_state = torch.cat([current_partial_state, current_joints], dim=-1) <--- 删除这行
            all_states[episode_start_idx:episode_start_idx + batch_size, t] = current_partial_state # <--- 直接存 Partial State
            
            # 4. Store Metadata
            all_masks[episode_start_idx:episode_start_idx + batch_size, t] = active_mask.cpu().float()
            
            # 5. Action
            random_action = torch.rand((batch_size, action_dim), device=device) * (action_high - action_low) + action_low
            all_actions[episode_start_idx:episode_start_idx + batch_size, t] = random_action.cpu()
            
            # Step environment
            obs, reward, term, trunc, info = env.step(random_action)
            
            all_rewards[episode_start_idx:episode_start_idx + batch_size, t] = reward.cpu()
            
            # Update active mask
            done = torch.logical_or(term, trunc)
            active_mask = active_mask & (~done)
    
    env.close()
    
    # =========================================================================
    # Trim and Save
    # =========================================================================
    all_obss = all_obss[:num_episodes]
    all_states = all_states[:num_episodes]
    all_joints = all_joints[:num_episodes]
    all_actions = all_actions[:num_episodes]
    all_rewards = all_rewards[:num_episodes]
    all_masks = all_masks[:num_episodes]
    
    save_dir = os.path.join(path_to_save_data, "sphinx_format")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"collect_{env_id}.pt")
    
    data_dict = {
        "obss": all_obss,       # (N, T, H, W, C) - RGB observations
        "states": all_states,   # (N, T, D) - State vector (Partial, NO Joints)
        "joints": all_joints,   # (N, T, J) - Joints
        "actions": all_actions, # (N, T, action_dim)
        "rewards": all_rewards, # (N, T)
        "masks": all_masks,     # (N, T) - 1 for valid, 0 for done
    }
    
    torch.save(data_dict, save_path)
    
    print(f"\n{'='*60}")
    print(f"Data saved to: {save_path}")
    print(f"{'='*60}")
    print(f"Data shapes:")
    print(f"  obss:    {all_obss.shape} (N_episodes, T, H, W, C) - RGB observations")
    print(f"  states:  {all_states.shape} (N_episodes, T, state_dim) - Full state vector")
    print(f"  joints:  {all_joints.shape} (N_episodes, T, joints_dim) - Robot joints")
    print(f"  actions: {all_actions.shape} (N_episodes, T, action_dim)")
    print(f"  rewards: {all_rewards.shape} (N_episodes, T)")
    print(f"  masks:   {all_masks.shape} (N_episodes, T)")
    print(f"{'='*60}")
    
    return save_path


@dataclass
class Args:
    env_id: str = "ShellGameTouch-v0"
    """Environment ID to collect data from"""
    path_to_save_data: str = "data"
    """Directory to save collected data"""
    num_episodes: int = 1000
    """Number of episodes to collect"""
    batch_size: int = 128
    """Batch size for parallel environment collection"""
    seed: int = 42
    """Random seed"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    print(f"\n{'='*60}")
    print(f"Random Data Collection for Sphinx VAE Training")
    print(f"{'='*60}")
    print(f"Environment: {args.env_id}")
    print(f"Num episodes: {args.num_episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save path: {args.path_to_save_data}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")
    
    collect_random_data_sphinx_format(
        env_id=args.env_id,
        path_to_save_data=args.path_to_save_data,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        seed=args.seed
    )
