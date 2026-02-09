import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyrallis
import numpy as np
import torch

from typing import Optional
from utils.logger import Logger
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

from env import utils_env
from utils.utils_reward import set_seed, collect_feedback, collect_human_feedback, consist_test_dataset, compute_mean_std, normalize_states
from models.reward_model import RewardModel


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cpu"
    dataset: str = "medium-expert"
    env: str = "metaworld_dial-turn-v2"  # environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    log_path: Optional[str] = "log"  # log path
    load_model: str = ""  # Model load file name, "" doesn't load
    # preference learning
    feedback_num: int = 1000
    data_quality: float = 2.0  # Replay buffer size (data_quality * 100000)
    segment_size: int = 25
    normalize: bool = True
    threshold: float = 0.0
    data_aug: str = "none"
    q_budget: int = 10000
    feedback_type: str = "RLT"
    model_type: str = "BT"
    noise: float = 0.0
    human: bool = False
    # MLP
    epochs: int = int(1e3)
    batch_size: int = 256
    activation: str = "tanh"  # Final Activation function
    lr: float = 1e-3
    hidden_sizes: int = 128
    ensemble_num: int = 3
    ensemble_method: str = "mean"

    def __post_init__(self):
        if self.dataset=="medium-replay":
            self.log_path = f"log/{self.env}/medium-replay/data_{self.data_quality}_fn_{self.feedback_num}_qb_{self.q_budget}_ft_{self.feedback_type}_m_{self.model_type}/s_{self.seed}"
        elif self.dataset=="medium-expert":
            self.log_path = f"log/{self.env}/medium-expert/fn_{self.feedback_num}/s_{self.seed}"
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)


@pyrallis.wrap()
def train(config: TrainConfig):
    if config.device==None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif config.device!=None and config.device.isdigit():
        assert torch.cuda.device_count()>int(config.device), "invalid device"
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{config.device}"
        config.device = "cuda"

    set_seed(config.seed)

    log_path = os.path.join(config.log_path, "reward_model")
    writer = SummaryWriter(log_path)
    logger = Logger(writer=writer,log_path=log_path)

    if config.dataset == "medium-replay":
        dataset = utils_env.MetaWorld_mr_dataset_fake_data_test(config)
    elif config.dataset == "medium-expert":
        dataset = utils_env.MetaWorld_me_dataset_fake_data_test(config)

    N = dataset["observations"].shape[0]
    traj_total = N // 500  # each trajectory has 500 steps
    print(f"dataset size: {N}, traj_total: {traj_total}")

    if config.normalize:
        state_mean, state_std = compute_mean_std(
            dataset["observations"], eps=0
        )
        state_std[state_std < 1e-12] = 1.0
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    assert config.q_budget >= 1
    if config.human == False:
        multiple_ranked_list = collect_feedback(dataset, traj_total, config)
        print(len(multiple_ranked_list))
    elif config.human == True:
        multiple_ranked_list = collect_human_feedback(dataset, config)
        print(len(multiple_ranked_list))

    idx_st_1 = []
    idx_st_2 = []
    labels = []
    # construct the preference pairs
    for single_ranked_list in multiple_ranked_list:
        sub_index_set = []
        for i, group in enumerate(single_ranked_list):
            for tup in group:
                sub_index_set.append((tup[0], i, tup[1]))
        for i in range(len(sub_index_set)):
            for j in range(i + 1, len(sub_index_set)):
                idx_st_1.append(sub_index_set[i][0])
                idx_st_2.append(sub_index_set[j][0])
                if sub_index_set[i][1] < sub_index_set[j][1]:
                    labels.append([0, 1])
                else:
                    labels.append([0.5, 0.5])
    labels = np.array(labels)
    idx_1 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_1]
    idx_2 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_2]
    obs_act_1 = np.concatenate(
        (dataset["observations"][idx_1], dataset["actions"][idx_1]), axis=-1
    )
    obs_act_2 = np.concatenate(
        (dataset["observations"][idx_2], dataset["actions"][idx_2]), axis=-1
    )
    # test query set (for debug the training, not used for training)
    test_feedback_num = 5000
    test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels = (
        consist_test_dataset(
            dataset,
            test_feedback_num,
            traj_total,
            segment_size=config.segment_size,
            threshold=config.threshold,
        )
    )

    dimension = obs_act_1.shape[-1]
    reward_model = RewardModel(config, obs_act_1, obs_act_2, labels, dimension, logger)

    reward_model.save_test_dataset(
        test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels
    )

    reward_model.train_model()
    reward_model.save_model(config.log_path)


if __name__ == "__main__":
    train()
