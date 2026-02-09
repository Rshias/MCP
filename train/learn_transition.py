import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pyrallis

from env import utils_env
from models.transition_model import TransitionModel
from offlinerlkit.utils.logger import Logger

@dataclass
class TrainConfig:
    # Experiment
    device: str = "cpu"
    dataset: str = "medium-replay"
    env: str = "metaworld_lever-pull-v2"
    seed: int = 0
    log_path: Optional[str] = "log"
    load_model: str = ""
    feedback_num: int = 1000
    data_quality: float = 5.0
    segment_size: int = 25
    normalize: bool = True
    threshold: float = 0.0
    data_aug: str = "none"
    q_budget: int = 10000
    feedback_type: str = "RLT"
    model_type: str = "BT"
    noise: float = 0.0
    human: bool = False
    epochs: int = int(500)
    batch_size: int = 256
    activation: str = "tanh"
    lr: float = 1e-3
    hidden_sizes: int = 200
    ensemble_num: int = 7
    num_elites: int = 5
    ensemble_method: str = "mean"

    def __post_init__(self):
        if self.log_path is None:
            if self.dataset == "medium-replay":
                self.log_path = (
                    f"log/{self.env}/medium-replay/fn_{self.feedback_num}"
                    f"_qb_{self.q_budget}_m_{self.model_type}"
                )


@pyrallis.wrap()
def train(config: TrainConfig):
    if config.device==None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif config.device!=None and config.device.isdigit():
        assert torch.cuda.device_count()>int(config.device), "invalid device"
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{config.device}"
        config.device = "cuda"
        
    dataset = utils_env.MetaWorld_mr_dataset(config)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True

    obs_dim = dataset["observations"].shape[1]
    act_dim = dataset["actions"].shape[1]

    model = TransitionModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=(config.hidden_sizes,) * 4,
        num_ensemble=config.ensemble_num,
        num_elites=config.num_elites,
        device=config.device
    )


    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(save_path, output_config)
    logger.log_hyperparameters(vars(config))
    
    

    if config.load_model and os.path.exists(config.load_model):
        model.load(config.load_model)
        print(f"model load：{config.load_model}")
    else:
        model.train(
            dataset=dataset,
            batch_size=config.batch_size,
            max_epochs=config.epochs,
            logvar_loss_coef=0.01,
            logger = logger
        )
        model.save(save_path)
        print(f"model save：{save_path}")


if __name__ == "__main__":
    train()
