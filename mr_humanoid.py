#!/usr/bin/env python3
# based on https://github.com/chwoong/LiRE
import os, sys, random, datetime, copy, math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pyrallis

# import gym
import gymnasium as gym  # changed
from tqdm import tqdm
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from env import utils_env
from utils.logger import Logger

sys.path.append("./Reward_learning")
from reward_learning import reward_model

TensorBatch = List[torch.Tensor]
TrajTensorBatch = List[List[torch.Tensor]]

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # Experiment
    device: str = None
    dataset: str = "medium-replay"
    env: str = "metaworld_box-close-v2"  # environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 50  # How many episodes run during evaluation
    max_timesteps: int = 250000  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    data_quality: float = None  # Replay buffer size (data_quality * 100000)
    trivial_reward: int = 0  # 0: GT reward, 1: zero reward, 2: constant reward, 3: negative reward
    # Algorithm
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = True  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # Reward model
    feedback_num: int = 1000
    use_reward_model: bool = True
    epochs: int = 0
    batch_size: int = 256
    activation: str = "tanh"
    lr: float = 1e-3
    threshold: float = 0.5
    segment_size: int = 25
    data_aug: str = "none"
    hidden_sizes: int = 128
    ensemble_num: int = 3
    ensemble_method: str = "mean"
    q_budget: int = 1
    feedback_type: str = "RLT"
    model_type: str = "BT"
    noise: float = 0.0
    human: bool = False

    def __post_init__(self):
        if self.dataset == "medium-replay":
            self.log_path = (
                f"log/{self.env}/medium-replay/data_{self.data_quality}_fn_{self.feedback_num}_"
                f"qb_{self.q_budget}_ft_{self.feedback_type}_m_{self.model_type}/s_{self.seed}"
            )
        elif self.dataset == "medium-expert":
            self.log_path = f"log/{self.env}/medium-expert/fn_{self.feedback_num}/s_{self.seed}"


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


@torch.no_grad()
def eval_metaworld(env, task_name, actor: nn.Module, device: str, n_episodes, seed):
    actor.eval()
    eval_scores, eval_success = [], []

    # action clipping (avoids MuJoCo NaNs)
    has_bounds = hasattr(env, "action_space") and hasattr(env.action_space, "low")
    if has_bounds:
        a_low = np.asarray(env.action_space.low, dtype=np.float32)
        a_high = np.asarray(env.action_space.high, dtype=np.float32)

    for ep in tqdm(range(n_episodes), desc="Evaluating"):
        try:
            obs, info = env.reset(seed=seed + ep)
        except TypeError:
            out = env.reset()
            obs = out[0] if isinstance(out, tuple) else out

        if isinstance(obs, tuple):
            obs = obs[0]

        done = False
        total_reward = 0.0
        success_flag = 0
        saw_success_key = False

        while not done:
            action = actor.act(obs, device)
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            if action.ndim == 2 and action.shape[0] == 1:
                action = action.squeeze(0)
            if has_bounds:
                action = np.clip(action, a_low, a_high)

            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = result

            if isinstance(obs, tuple):
                obs = obs[0]

            total_reward += float(reward)

            if isinstance(info, dict) and "success" in info:
                saw_success_key = True
                if info.get("success", False):
                    success_flag = 1
                    break

        eval_scores.append(total_reward)
        eval_success.append(success_flag if saw_success_key else 0)

    actor.train()
    return np.array(eval_scores, dtype=np.float32), np.array(eval_success, dtype=np.int32)


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    state_mean = np.asarray(state_mean, dtype=np.float32)
    state_std = np.asarray(state_std, dtype=np.float32)
    eps = 1e-3

    def normalize_state(state):
        if isinstance(state, tuple) and len(state) == 2:
            state = state[0]
        state = np.asarray(state, dtype=np.float32)
        return (state - state_mean) / (state_std + eps)

    env = gym.wrappers.TransformObservation(
        env,
        normalize_state,
        observation_space=env.observation_space,
    )

    if reward_scale != 1.0:
        def scale_reward(reward):
            return reward_scale * reward

        env = gym.wrappers.TransformReward(env, scale_reward)

    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions}")

    def load_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        # self.num_traj = n_transitions // 500
        self.num_traj = n_transitions // 1000
        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def sample_trajectory(self, batch_size: int, segment_size: int) -> TensorBatch:
        traj_idx = np.random.choice(self.num_traj, batch_size, replace=True)
        idx_start = [1000 * i + np.random.randint(0, 1000 - segment_size) for i in traj_idx]
        indices = []
        for i in idx_start:
            indices.extend([j for j in range(i, i + segment_size)])
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        return [states, actions, rewards, next_states]


def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if env is None:
        return

    try:
        env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)

    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)

    if hasattr(env, "envs"):
        for e in env.envs:
            try:
                e.reset(seed=seed)
            except TypeError:
                if hasattr(e, "seed"):
                    e.seed(seed)
            if hasattr(e, "action_space") and hasattr(e.action_space, "seed"):
                e.action_space.seed(seed)
            if hasattr(e, "observation_space") and hasattr(e.observation_space, "seed"):
                e.observation_space.seed(seed)


@torch.no_grad()
def eval_actor(
    env: gym.Env,
    env_name: str,
    actor: nn.Module,
    device: str,
    n_episodes: int,
    seed: int,
) -> np.ndarray:
    actor.eval()
    episode_rewards = []
    episode_success_list = []
    for ep in range(n_episodes):
        try:
            state, info = env.reset(seed=seed + ep)
        except TypeError:
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
        done = False

        episode_reward = 0.0
        episode_success = 0
        while not done:
            action = actor.act(state, device)
            step_out = env.step(action)
            if len(step_out) == 5:
                state, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                state, reward, done, info = step_out
            episode_reward += float(reward)
            if "metaworld" in env_name and isinstance(info, dict) and "success" in info:
                episode_success = max(episode_success, int(info["success"]))

        episode_rewards.append(episode_reward)
        episode_success_list.append(episode_success)

    actor.train()
    return np.array(episode_rewards), np.array(episode_success_list)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, max_episode_steps=1000, trivial_reward=0):
    if trivial_reward == 0:
        dataset["rewards"] = (dataset["rewards"] - min(dataset["rewards"])) / (
            max(dataset["rewards"]) - min(dataset["rewards"])
        )
    elif trivial_reward == 1:
        dataset["rewards"] *= 0.0
    elif trivial_reward == 2:
        dataset["rewards"] = (dataset["rewards"] - min(dataset["rewards"])) / (
            max(dataset["rewards"]) - min(dataset["rewards"])
        )
        min_reward, max_reward = min(dataset["rewards"]), max(dataset["rewards"])
        dataset["rewards"] = np.random.uniform(
            min_reward, max_reward, size=dataset["rewards"].shape
        )
    elif trivial_reward == 3:
        dataset["rewards"] = 1 - (dataset["rewards"] - min(dataset["rewards"])) / (
            max(dataset["rewards"]) - min(dataset["rewards"])
        )


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP([state_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = float(max_action)

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)  # unsquashed
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)                # Normal in R
        z = dist.mean if not self.training else dist.sample()
        a = torch.tanh(z) * self.max_action
        return a.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            dropout=dropout,
        )
        self.max_action = float(max_action)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state_t = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        z = self(state_t)
        a = torch.tanh(z) * self.max_action
        return a.cpu().data.numpy().flatten()


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


def atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class MR:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = float(max_action)
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["train/value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        traj_batch: TrajTensorBatch,
        log_dict: Dict,
    ):
        # NOTE: original code ignores terminals for MetaWorld; keep behavior
        targets = rewards + self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        reg_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        q_loss = reg_loss
        log_dict["train/reg_loss"] = reg_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):

        max_a = self.max_action
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)

        policy_out = self.actor(observations)  
        if isinstance(policy_out, torch.distributions.Distribution):
          
            a_unit = (actions / max_a).clamp(-1 + 1e-6, 1 - 1e-6)
          
            z = atanh(a_unit) 
      
            logp_base = policy_out.log_prob(z).sum(dim=-1, keepdim=False)  
           
            log_jac_tanh = torch.log(1 - a_unit.pow(2) + 1e-6).sum(dim=-1, keepdim=False)
            scale_term = actions.shape[-1] * math.log(max_a + 1e-6)
            logp_env = logp_base - log_jac_tanh - scale_term
            bc_losses = -logp_env  
        elif torch.is_tensor(policy_out):
           
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape mismatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError

        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["train/actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch, traj_batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        observations, actions, rewards, next_observations, dones = batch
        log_dict = {}
        with torch.no_grad():
            next_v = self.vf(next_observations)
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        self._update_q(next_v, observations, actions, rewards, dones, traj_batch, log_dict)
        self._update_policy(adv, observations, actions, log_dict)
        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])
        self.total_it = state_dict["total_it"]


@torch.no_grad()
def eval_and_record(env, actor, device, step, n_episodes, logger, config):
    scores, success = eval_metaworld(
        env, task_name="metaworld", actor=actor, device=device,
        n_episodes=n_episodes, seed=config.seed
    )
    mean_score = float(scores.mean())
    success_rate = float(success.mean())
    logger.record("eval/eval_score", mean_score, step)
    logger.record("eval/eval_success", success_rate * 100.0, step)

    csv_path = os.path.join(config.log_path, f"appo_{config.env}.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["step", "mean_score", "success_rate"])
        w.writerow([step, mean_score, success_rate])

    return mean_score, success_rate


@pyrallis.wrap()
def train(config: TrainConfig):
    if config.device is None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif config.device is not None and config.device.isdigit():
        assert torch.cuda.device_count() > int(config.device), "invalid device"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{config.device}"
        config.device = "cuda"

    log_path = os.path.join(config.log_path, f"MR")
    writer = SummaryWriter(log_path)
    logger = Logger(writer=writer, log_path=log_path)

    
    env = gym.make("Humanoid-v5")
   #  env = gym.make("Ant-v5")

    if config.dataset == "medium-replay":
        dataset = utils_env.MetaWorld_mr_dataset(config)
    elif config.dataset == "medium-expert":
        dataset = utils_env.MetaWorld_me_dataset(config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dimension = state_dim + action_dim

    seed = config.seed
    set_seed(seed, env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)

    if config.use_reward_model:
        model = reward_model.RewardModel(config, None, None, None, dimension, None)
        model.load_model(config.log_path)
        dataset["rewards"] = model.get_reward(dataset)
        print("labeled by reward model")

    if config.normalize_reward:
        modify_reward(dataset, max_episode_steps=500, trivial_reward=config.trivial_reward)

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    # Action bound from env (e.g., Humanoid-v5: 0.4)
    max_action = float(getattr(env.action_space, "high", np.array([0.4]))[0])

    
    config.buffer_size = dataset["observations"].shape[0]
    replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    replay_buffer.load_dataset(dataset)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout)
        if config.iql_deterministic
        else GaussianPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout)
    ).to(config.device)

    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    trainer = MR(**kwargs)

    eval_steps, eval_scores_hist, eval_success_hist = [], [], []
    for t in tqdm(range(int(config.max_timesteps)), desc="Training (offline IQL)"):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        traj_batch = replay_buffer.sample_trajectory(config.batch_size, config.segment_size)
        traj_batch = [b.to(config.device) for b in traj_batch]
        log_dict = trainer.train(batch, traj_batch)

        if (t + 1) % 5000 == 0:
            for k, v in log_dict.items():
                logger.record(k, v, trainer.total_it)

        if (t + 1) % config.eval_freq == 0:
            mean_score, success_rate = eval_and_record(
                env=env,
                actor=actor,
                device=config.device,
                step=trainer.total_it,
                n_episodes=config.n_episodes,
                logger=logger,
                config=config,
            )
            eval_steps.append(trainer.total_it)
            eval_scores_hist.append(mean_score)
            eval_success_hist.append(success_rate)


if __name__ == "__main__":
    train()
