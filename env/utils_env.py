import numpy as np
import torch
import torch.nn.functional as F
import gym
import os

import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from gym.wrappers.time_limit import TimeLimit
from env.wrappers import NormalizedBoxEnv
import pickle as pkl


def make_metaworld_env(env_name, seed):
    env_name = env_name.replace("metaworld_", "")
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls()
    # print("partially observe", env._partially_observable) Ture
    # print("env._freeze_rand_vec", env._freeze_rand_vec) True
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)
    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)


def MetaWorld_mr_dataset(config):
    """
    MetaWorld medium-replay dataset from LiRE (Choi et al., 2024)
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if config.human == False:
        base_path = os.path.join(os.getcwd(), "dataset/MetaWorld/")
        env_name = config.env
        base_path += str(env_name.replace("metaworld_", ""))
        dataset = dict()
        for seed in range(3):
            path = base_path + f"/saved_replay_buffer_1000000_seed{seed}.pkl"
            with open(path, "rb") as f:
                load_dataset = pkl.load(f)
            for key in load_dataset.keys():
                load_dataset[key] = load_dataset[key][
                    : int(config.data_quality * 100_000)
                ]
            load_dataset["terminals"] = load_dataset["dones"][
                : int(config.data_quality * 100_000)
            ]
            load_dataset.pop("dones", None)
            for key in load_dataset.keys():
                if key not in dataset:
                    dataset[key] = load_dataset[key]
                else:
                    dataset[key] = np.concatenate(
                        (dataset[key], load_dataset[key]), axis=0
                    )
    elif config.human == True:
        base_path = os.path.join(os.getcwd(), "human_feedback/")
        base_path += f"{config.env}/dataset.pkl"
        with open(base_path, "rb") as f:
            dataset = pkl.load(f)
            dataset["observations"] = np.array(dataset["observations"])
            dataset["actions"] = np.array(dataset["actions"])
            dataset["next_observations"] = np.array(dataset["next_observations"])
            dataset["rewards"] = np.array(dataset["rewards"])
            dataset["terminals"] = np.array(dataset["dones"])

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    dataset["rewards"] = dataset["rewards"].reshape(-1)
    dataset["terminals"] = dataset["terminals"].reshape(-1)

    for i in range(N):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["next_observations"][i].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }


def MetaWorld_me_dataset(config):
    """
    MetaWorld medium-expert dataset following the approaches of IPL (Hejna & Sadigh, 2024) and LiRE (Choi et al., 2024)
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    base_path = os.path.join(os.getcwd(), "dataset/MetaWorld_medium-expert/" + str(config.env).split("_")[1])
    load_dataset = np.load(os.path.join(base_path, "trajectory.npz"))
    dataset = {key: load_dataset[key] for key in load_dataset.keys()}

    if config.data_quality * 100_000 >= dataset["rewards"].shape[0]:
        idx = np.arange(dataset["rewards"].shape[0]//500)
    else:
        N = dataset["rewards"].shape[0] // 500  # 600 for metaworld medium-expert
        # take trajectories proportional to the data quality
        n_expert = int(config.data_quality * 200 / 12)
        idx_expert = np.arange(n_expert)
        n_within_env = int(config.data_quality * 200 / 12)
        idx_within_env = np.arange(n_within_env) + int(N/12)
        n_random = int(config.data_quality * 200 / 3)
        idx_random = np.arange(n_random) + int(N/6)
        n_eps_greedy = int(config.data_quality * 200 / 3)
        idx_eps_greedy = np.arange(n_eps_greedy) + int(N/3)
        n_cross_env = int(config.data_quality * 200 / 6)
        idx_cross_env = np.arange(n_cross_env) + int(2*N/3)

        idx = np.concatenate((idx_expert, idx_within_env, idx_random, idx_eps_greedy, idx_cross_env), axis=0)

    state_dim = dataset["states"].shape[1]
    action_dim = dataset["actions"].shape[1]

    return {
        "observations": dataset["states"].astype(np.float32).reshape(-1,500,state_dim)[idx].reshape(-1,state_dim),
        "actions": dataset["actions"].astype(np.float32).reshape(-1,500,action_dim)[idx].reshape(-1,action_dim),
        "next_observations": dataset["next_states"].astype(np.float32).reshape(-1,500,state_dim)[idx].reshape(-1,state_dim),
        "rewards": dataset["rewards"].astype(np.float32).reshape(N,500,-1)[idx].reshape(-1),
        "terminals": dataset["dones"].astype(bool).reshape(N,500,-1)[idx].reshape(-1),
    }


