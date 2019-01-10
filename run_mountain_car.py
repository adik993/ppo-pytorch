import numpy as np
import torch

from agents import PPO
from curiosity import ICM, MlpICMModel
from envs import MultiEnv
from models import MLP
from reporters import TensorBoardReporter
from rewards import GeneralizedAdvantageEstimation, GeneralizedRewardEstimation

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reporter = TensorBoardReporter()

    agent = PPO(MultiEnv('MountainCar-v0', 16, reporter=reporter),
                reporter=reporter,
                normalize_state=True,
                normalize_reward=True,
                model_factory=MLP.factory(),
                curiosity_factory=ICM.factory(MlpICMModel.factory(), policy_weight=1, reward_scale=0.01, weight=0.2,
                                              intrinsic_reward_integration=0.01, reporter=reporter),
                # curiosity_factory=NoCuriosity.factory(),
                reward=GeneralizedRewardEstimation(gamma=0.99, lam=0.95),
                advantage=GeneralizedAdvantageEstimation(gamma=0.99, lam=0.95),
                learning_rate=2e-3,
                clip_range=0.2,
                v_clip_range=0.2,
                c_entropy=1e-2,
                c_value=0.5,
                n_mini_batches=32,
                n_optimization_epochs=3,
                clip_grad_norm=0.5)
    agent.to(device, torch.float32, np.float32)

    agent.learn(epochs=50, n_steps=256)
    agent.eval(n_steps=600, render=True)
