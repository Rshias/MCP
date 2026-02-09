
## Dependencies

Install pacakges with `requirements.txt` file
```
conda create -n pbrl python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tensorboard ipykernel matplotlib seaborn
pip install "gym[mujoco_py,classic_control]==0.23.0"
pip install pyrallis tqdm
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
pip install -r requirements.txt
```

## Datasets
Meta-world `medium-replay` dataset is available in the official repository of [LiRE](https://github.com/chwoong/LiRE).  

## Training
Set learning rates, network architectures, batch sizes, and other algorithmic hyperparameter by modifying config files.

To train reward model:
```
python train/learn_reward.py --config=configs/medium-replay/task-name-v2/reward.yaml
```

```
To train transition model, 
```
python train/learn_transition.py--config=configs/medium-replay/task-name-v2/transition.yaml
```

To run PbRL algorithm,
```
python main.py --config=configs/medium-replay/task-name-v2/pbrl.yaml
```

## Results
The training results are stored in `log/`.

## Reference

Our code is based on the official implementation of \<APPO : Adversarial Preference-based Policy Optimization\> : [https://github.com/oh-lab/APPO](https://github.com/oh-lab/APPO) 
