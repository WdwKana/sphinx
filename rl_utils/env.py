import gym
import gym_minigrid
import escape_room

def make_env(env_key, seed=None):
    # These legacy environments predate Gym's stricter checker and may
    # return extra observation keys like "mission" or "direction".
    try:
        env = gym.make(env_key, disable_env_checker=True)
    except TypeError:
        env = gym.make(env_key)
    env.seed(seed)
    return env

def make_env_with_env_name(args_env, args_procs, seed=None):
    envs = []
    for i in range(args_procs):
        envs.append(make_env(args_env, seed + 10000 * i))
    return envs