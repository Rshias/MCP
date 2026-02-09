from copy import deepcopy
import os, sys, random, datetime
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pyrallis
from tqdm import tqdm
import copy
import gym
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import metaworld
from env import utils_env
from utils.logger import Logger
from models.reward_model import RewardModel

from models.transition_model import TransitionModel
import csv
from pathlib import Path



from models.base_models import MLP, ActorProb, Critic, DiagGaussian

TensorBatch = List[torch.Tensor]
TrajTensorBatch = List[List[torch.Tensor]]

@dataclass
class TrainConfig:
    device: str = None
    dataset: str = "medium-replay"
    env: str = "metaworld_box-close-v2"
    seed: int = 0
    eval_freq: int = int(5e3)
    n_episodes: int = 100
    max_timesteps: int = 1e6
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    data_quality: float = None
    trivial_reward: int = 0
    buffer_size: int = 2_000_000
    batch_size: int = 256
    traj_batch_size: int = 16
    discount: float = 0.99
    normalize: bool = True
    normalize_reward: bool = True
    actor_lr: float = 1e-4
    actor_ref_lr: float = 3e-4
    hidden_size: int = 256
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
    lambda_1: float = 1
    lambda_2: float = 1
    eta: float = 0.1
    reward_lr: float = 3e-4
    transition_lr: float = 3e-4
    pretrain_epoch: int = 10
    rollout_length: int = 2
    model_update_period: int = 50
    pretrain: bool = False
    def __post_init__(self):
        if self.dataset == "medium-replay":
            self.log_path = f"log/{self.env}/medium-replay/data_{self.data_quality}_fn_{self.feedback_num}_qb_{self.q_budget}_ft_{self.feedback_type}_m_{self.model_type}/s_{self.seed}"
        elif self.dataset == "medium-expert":
            self.log_path = f"log/{self.env}/medium-expert/fn_{self.feedback_num}/s_{self.seed}"

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

def wrap_env(
        env: gym.Env,
        state_mean: Union[np.ndarray, float] = 0.0,
        state_std: Union[np.ndarray, float] = 1.0,
        reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        if len(state) == 2:
            state = state[0]
        return (
                state - state_mean
        ) / state_std
    def scale_reward(reward):
        return reward_scale * reward
    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env



@torch.no_grad()
def eval_metaworld(env, task_name, agent, n_episodes, seed):
    agent.eval()
    eval_scores = []
    eval_success = []
    for ep in tqdm(range(n_episodes), desc="Evaluating"):
        obs, done = env.reset(), False
        if isinstance(obs, tuple):
            obs = obs[0]
        total_reward = 0
        success_flag = False
        while not done:
            action = agent.sample_action(obs, deterministic=False)
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            if action.ndim == 2 and action.shape[0] == 1:
                action = action.squeeze(0)
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            if isinstance(obs, tuple):
                obs = obs[0]
            total_reward += reward
            if info.get("success", False):
                success_flag = 1
                break
        eval_scores.append(total_reward)
        eval_success.append(success_flag)
    agent.train()
    return np.array(eval_scores), np.array(eval_success)

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

    def add(self, state, action, reward, done, next_state):
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)
        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

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
        rewards = data["rewards"]
        if rewards.ndim == 1:
            rewards = rewards[:, None]
        elif rewards.ndim == 3:
            rewards = rewards.squeeze(-1)
        self._rewards[:n_transitions] = self._to_tensor(rewards)
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        dones = data["terminals"]
        if dones.ndim == 1:
            dones = dones[:, None]
        elif dones.ndim == 3:
            dones = dones.squeeze(-1)
        self._dones[:n_transitions] = self._to_tensor(dones)
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        self.num_traj = n_transitions // 500
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
        # traj_idx = np.random.choice(self.num_traj, batch_size, replace=True)
        # idx_start = [500 * i + np.random.randint(0, 499 - segment_size) for i in traj_idx]
        
        # pick trajectories and starts on GPU
        traj_ids = torch.randint(self.num_traj, (batch_size,), device=self._device)
        starts   = 500 * traj_ids + torch.randint(0, 500 - segment_size, (batch_size,), device=self._device)
       
        H   = segment_size
        rel = torch.arange(H, device=self._device).unsqueeze(0)         
        idx = (starts.unsqueeze(1) + rel).reshape(-1).long()             

        states      = self._states.index_select(0, idx).view(batch_size, H, -1)
        actions     = self._actions.index_select(0, idx).view(batch_size, H, -1)
        rewards     = self._rewards.index_select(0, idx).view(batch_size, H, -1)
        next_states = self._next_states.index_select(0, idx).view(batch_size, H, -1)
        return [states, actions, rewards, next_states]


def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)



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


def modify_reward(
        dataset,
        max_episode_steps=1000,
        trivial_reward=0,
):
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

class PbRL:
    def __init__(
            self,
            actor,
            actor_old,
            actor_ref,
            reward_model,
            transition_model,
            reward_model_mle,
            transition_model_mle,
            reward_optim,
            transition_optim,
            lambda_1,
            lambda_2,
            eta,
            actor_optim,
            actor_ref_optim,
            device,
            gamma=0.99,
            batch_size=256,
            traj_batch_size=16,
            max_steps=250000,
            state_mean=0,
            state_std=1,
            bc_coef: float = 3,
            kl_coef: float = 10,
            model_update_period: int = 10,
            rollout_length: int = 2
    ):
        super().__init__()
        self.actor = actor
        self.actor_old = actor_old
        self.actor_ref = actor_ref
        self.actor_optim = actor_optim
        self.actor_ref_optim = actor_ref_optim
        self.device = device
        self.transition_model = transition_model
        self.reward_model_mle = reward_model_mle
        self.transition_model_mle = transition_model_mle
        self.reward_optim = reward_optim
        self.transition_optim = transition_optim
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.eta = eta
        self.reward_model = reward_model
        self._gamma = gamma
        self.batch_size = batch_size
        self.traj_batch_size = traj_batch_size
        self.__eps = np.finfo(np.float32).eps.item()
        self._device = device
        self.total_it = 0
        self.state_mean = torch.tensor(state_mean, device = self._device)
        self.state_std = torch.tensor(state_std, device = self._device)
        self.rollout_length = rollout_length

    
    
    def train(self):
        self.actor.train()

    def eval(self):
        self.actor.eval()

    def __call__(self, obs, deterministic=False):
        dist = self.actor.get_dist(obs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action)
        action_scale = torch.tensor((1 - (-1)) / 2, device=action.device)
        squashed_action = torch.tanh(action)
        log_prob = log_prob - torch.log(action_scale * (1 - squashed_action.pow(2)) + self.__eps).sum(-1, keepdim=True)
        
        return squashed_action, log_prob

    def log_prob(self, obs, action):
        dist = self.actor.get_dist(obs)
        log_prob = dist.log_prob(action)
        action_scale = torch.tensor((self.action_space.high - self.action_space.low) / 2, device=action.device)
        log_prob = log_prob - torch.log(action_scale * (1 - action.pow(2)) + self.__eps).sum(-1, keepdim=True)
        return log_prob

    def sample_action(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action = self(obs, deterministic)[0]
        return action.cpu().detach().numpy()
    

        
    def learn(self, batch: TensorBatch, traj_batch: TensorBatch) -> Dict[str, float]:
  
        self.total_it += 1
        
        if (self.total_it+1) % 2 == 0:  
            self.actor_old.load_state_dict(self.actor.state_dict())
        
        obs_MS, actions, rewards, next_obs, terminals = batch
        obs_t, action_t, reward_t, next_obs_t = traj_batch          
        
        B, H, D = obs_t.shape
        _, _, D_a = action_t.shape
        assert B % 2 == 0
        half = B // 2
        gamma = self._gamma

        S_list, LogP_list, R_list = [], [], []
        A_list = []
        
        obs = obs_t[:, 0, :]                                       

        J_vec = torch.zeros(B, device=self.device)     
        disc  = torch.ones(B, device=self.device)       
        for t in range(self.rollout_length):
            S_t = obs                                              

            S_t_actor = S_t.detach() 
            
            a_t, logp_t = self(S_t_actor, deterministic=False) 
           
            sa = torch.cat([S_t, a_t.detach()], dim=-1)
            r_t = self.reward_model.ensemble_model_forward(sa)      
            if r_t.dim() == 2:
                r_t = r_t.squeeze(-1)
        
            J_vec = J_vec + disc * r_t

            mean_mle, logvar = self.transition_model.model(sa)       
            std = torch.sqrt(torch.exp(logvar))
            delta = mean_mle.mean(dim=0)                           
            next_obs_real = S_t * self.state_std + self.state_mean + delta 
            obs = (next_obs_real - self.state_mean) / self.state_std

            S_list.append(S_t)
            LogP_list.append(logp_t.squeeze(-1))                   
            R_list.append(r_t.detach())                             
            A_list.append(a_t.detach())
            
            
        
        
        S = torch.stack(S_list,    0)                         
        logp = torch.stack(LogP_list, 0)                         
        rewards = torch.stack(R_list,    0)                        

        A = torch.stack(A_list, 0)                 
        
        
        mask = torch.ones_like(rewards, dtype=torch.bool, device=self.device)
        
        T = rewards.shape[0]
       
        J_pi = J_vec.mean() 

        # returns-to-go Rt[t] = r_t + γ Rt[t+1]
        G  = torch.zeros(B, dtype=torch.float32, device=self.device)
        Rt = torch.zeros_like(rewards)                              # [T,B]
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            Rt[t] = G

        # flatten valid steps
        mask_flat = mask.view(-1)
        S_flat    = S.view(T*B, -1)[mask_flat]                      
        logp_flat = logp.view(-1)[mask_flat]                       
        Rt_flat   = Rt.view(-1)[mask_flat]                          
        A_flat = A.view(-1, A.shape[-1])[mask_flat]  
        
        with torch.no_grad():
            adv = (Rt_flat - Rt_flat.mean()) / (Rt_flat.std() + 1e-8)
        adv = adv.detach()


        obs_act_BH = torch.cat([obs_t, action_t], dim=-1)   
        BH = obs_act_BH.shape[0] * obs_act_BH.shape[1]
        last_dim = obs_act_BH.shape[-1]
        obs_act_flat = obs_act_BH.reshape(BH, last_dim)                   
        R_all = self.reward_model.ensemble_model_forward(obs_act_flat)    
        if R_all.dim() == 2 and R_all.size(-1) == 1:
            R_all = R_all.squeeze(-1)                                     
        R_all = R_all.reshape(B, H)                                       

        with torch.no_grad():
            R_all_mle = self.reward_model_mle.ensemble_model_forward(obs_act_flat) 
            if R_all_mle.dim() == 2 and R_all_mle.size(-1) == 1:
                R_all_mle = R_all_mle.squeeze(-1)
            R_all_mle = R_all_mle.reshape(B, H)     

        traj_r = R_all.sum(dim=1).mean()                                  

        sum_per_traj = R_all.sum(dim=1).view(2, half)                  
        sum_per_traj_mle = R_all_mle.sum(dim=1).view(2, half)              

        delta = sum_per_traj[1] - sum_per_traj[0]                      
        delta_mle = sum_per_traj_mle[1] - sum_per_traj_mle[0]              
        E_1 = (delta - delta_mle).abs().mean()                             

    

    
        next_obs_flat = next_obs_t.reshape(B * H, D)*self.state_std + self.state_mean
        mean, logvar = self.transition_model.model(obs_act_flat)
        std  = torch.sqrt(torch.exp(logvar)) 
        with torch.no_grad():
            mean_mle, logvar_mle = self.transition_model_mle.model(obs_act_flat)
            std_mle = torch.sqrt(torch.exp(logvar_mle))



    
        dist = torch.distributions.Normal(mean, std)
        dist_mle = torch.distributions.Normal(mean_mle, std_mle)

      
        kl_new_old = torch.distributions.kl.kl_divergence(dist, dist_mle).sum(-1)
        E_2 = (kl_new_old).mean()

        total_loss = J_pi - traj_r + self.lambda_1 * E_1  + self.lambda_2 * E_2
    
        if self.total_it % self.model_update_period == 0
            self.reward_optim.zero_grad()
            self.transition_optim.zero_grad()
            total_loss.backward()
            self.reward_optim.step()
            self.transition_optim.step()

        
        S_flat_det = S_flat.detach()

        S_flat = S_flat.detach()
        dist_new = self.actor.get_dist(S_flat)                
        pg_loss = -(adv * logp_flat).mean()
        
        dist_ref = self.actor_ref.get_dist(obs_MS)
        kl_pi_ref = torch.distributions.kl.kl_divergence(dist, dist_ref).sum(-1).mean()
        
        dist = self.actor.get_dist(obs_MS)
        pred_actions = dist.rsample()
        mirror_loss = ((pred_actions - actions) ** 2).mean() 
        actor_loss = self.eta*mirror_loss + pg_loss + self.kl_coef * kl_pi_ref 

        self.actor_optim.zero_grad() 
        actor_loss.backward() 
        self.actor_optim.step()

        
        
       







@pyrallis.wrap()
def train(config: TrainConfig):
    if config.device is None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif config.device.isdigit():
        assert torch.cuda.device_count() > int(config.device)
        os.environ['CUDA_VISIBLE_DEVICES'] = config.device
        config.device = "cuda"
   
    writer = SummaryWriter(log_path)
    env = utils_env.make_metaworld_env(config.env, config.seed)
    if config.dataset == "medium-replay":
        dataset = utils_env.MetaWorld_mr_dataset(config)
    elif config.dataset == "medium-expert":
        dataset = utils_env.MetaWorld_me_dataset(config)

    # Modify the state and reward data
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    set_seed(config.seed, env)
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=0)
        state_std[state_std < 1e-12] = 1.0
    else:
        state_mean, state_std = 0, 1
    dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    
    # Initialize the replay buffer
    config.buffer_size = dataset["observations"].shape[0]
    replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    replay_buffer.load_dataset(dataset)
    
    # Initialize the actor networks
    actor_backbone = MLP(input_dim=state_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(latent_dim=actor_backbone.output_dim, output_dim=action_dim, unbounded=True, conditioned_sigma=True)
    actor = ActorProb(actor_backbone, dist, config.device)
    actor_old = copy.deepcopy(actor)

    actor_backbone_ref = MLP(input_dim=state_dim, hidden_dims=[256, 256])
    dist_ref = DiagGaussian(latent_dim=actor_backbone_ref.output_dim, output_dim=action_dim, unbounded=True, conditioned_sigma=True)
    actor_ref = ActorProb(actor_backbone_ref, dist_ref, config.device)

    if getattr(config, "pretrain_steps", 0) > 0:
        bc_optimizer = torch.optim.Adam(actor_ref.parameters(), lr=config.actor_ref_lr)
        actor_ref.train()
        num_samples = replay_buffer._size
        batch_size = config.batch_size
        for _ in range(config.pretrain_steps):
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                s_batch = replay_buffer._states[start:end]
                a_batch = replay_buffer._actions[start:end]

                dist_ref_batch = actor_ref.get_dist(s_batch)
                pred_actions_ref = dist_ref_batch.mode()
                bc_loss = F.mse_loss(pred_actions_ref, a_batch)

                bc_optimizer.zero_grad()
                bc_loss.backward()
                bc_optimizer.step()
    
    
    
    reward_model = RewardModel(config, None, None, None, state_dim+action_dim, None)

    reward_model.load_model(config.log_path)  # test
    for m in reward_model.ensemble_model:
        m.train()
    
    
    reward_model_mle = RewardModel(config, None, None, None, state_dim+action_dim, None)
    reward_model_mle.load_model(config.log_path)
    for m in reward_model_mle.ensemble_model: 
        for p in m.parameters(): p.requires_grad = False
        m.eval()

    transition_model = TransitionModel(obs_dim=state_dim, act_dim=action_dim, hidden_dims=(200, 200, 200, 200), num_ensemble=7, num_elites=5, device=config.device)
    transition_model.load(config.log_path)
    transition_model.model.train()


    transition_model_mle = TransitionModel(obs_dim=state_dim, act_dim=action_dim, hidden_dims=(200, 200, 200, 200), num_ensemble=7, num_elites=5, device=config.device)
    transition_model_mle.load(config.log_path)
    for p in transition_model_mle.model.parameters(): p.requires_grad = False
    transition_model_mle.model.eval()

    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    actor_ref_optim = None
    reward_optim = torch.optim.Adam(params=[p for model in reward_model.ensemble_model for p in model.parameters()], lr=config.reward_lr)
    transition_optim = transition_model.optim


    trainer = PbRL(
        actor=actor,
        actor_old=actor_old,
        actor_ref = actor_ref,
        actor_ref_optim=actor_ref_optim,
        reward_model=reward_model,
        transition_model=transition_model,
        model_update_period = config.model_update_period,
        reward_model_mle=reward_model_mle,
        transition_model_mle=transition_model_mle,
        actor_optim=actor_optim,
        reward_optim=reward_optim,
        transition_optim=transition_optim,
        lambda_1=config.lambda_1,
        lambda_2=config.lambda_2,
        eta=config.eta,
        device=config.device,
        state_mean = state_mean, 
        state_std = state_std,
        bc_coef = config.bc_coef,
        kl_coef = config.kl_coef,
        rollout_length = config.rollout_length,
        model_update_period = config.model_update_period,
    )
    trainer.train()
    eval_steps, eval_scores_hist, eval_success_hist = [], [], []

    for t in tqdm(range(int(config.max_timesteps))):
        
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        
        traj_batch = replay_buffer.sample_trajectory(2 * config.traj_batch_size, config.segment_size)
        
        traj_batch = [b.to(config.device) for b in traj_batch]
        log_dict = trainer.learn(batch, traj_batch)
        
    
        if (t + 1) % config.eval_freq == 0:
            mean_score, success_rate = eval_metaworld(env, task_name=config.env, agent=trainer, n_episodes=config.n_episodes, seed=config.seed)
            eval_steps.append(trainer.total_it)
            eval_scores_hist.append(mean_score)
            eval_success_hist.append(success_rate)
    

if __name__ == "__main__":
    train()
