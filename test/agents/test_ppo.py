import numpy as np
import pytest
import torch

from agents import PPO
from curiosity import ICM, MlpICMModel, NoCuriosity
from envs import MultiEnv
from models import MLP
from rewards import GeneralizedRewardEstimation, GeneralizedAdvantageEstimation


def agent_continuous():
    agent = PPO(MultiEnv('Pendulum-v0', 10),
                normalize_state=True,
                normalize_reward=True,
                model_factory=MLP.factory(),
                curiosity_factory=ICM.factory(MlpICMModel.factory(), policy_weight=1, reward_scale=0.01, weight=0.2,
                                              intrinsic_reward_integration=0.01),
                # curiosity_factory=NoCuriosity.factory(),
                reward=GeneralizedRewardEstimation(gamma=0.95, lam=0.1),
                advantage=GeneralizedAdvantageEstimation(gamma=0.95, lam=0.1),
                learning_rate=4e-4,
                clip_range=0.3,
                v_clip_range=0.3,
                c_entropy=1e-2,
                c_value=0.5,
                n_mini_batches=32,
                n_optimization_epochs=10,
                clip_grad_norm=0.5)
    agent.to(torch.device('cpu'), torch.float32, np.float32)
    return agent


def agent_discrete():
    agent = PPO(MultiEnv('CartPole-v1', 4),
                normalize_state=False,
                normalize_reward=False,
                model_factory=MLP.factory(),
                curiosity_factory=NoCuriosity.factory(),
                reward=GeneralizedRewardEstimation(gamma=0.99, lam=0.95),
                advantage=GeneralizedAdvantageEstimation(gamma=0.99, lam=0.95),
                learning_rate=5e-3,
                clip_range=0.2,
                v_clip_range=0.2,
                c_entropy=1e-2,
                c_value=0.5,
                n_mini_batches=4,
                n_optimization_epochs=5,
                clip_grad_norm=0.5)
    agent.to(torch.device('cpu'), torch.float32, np.float32)
    return agent


@pytest.mark.parametrize("agent", [
    (agent_continuous()),
    (agent_discrete())
], ids=["continuous", "discrete"])
def test_agent_learns_without_errors(agent):
    agent.learn(epochs=1, n_steps=10, initialization_steps=1, render=False)
    agent.eval(n_steps=10, render=False)
