from rewards.advantage import Advantage
from rewards.n_step_reward import NStepReward
import numpy as np


class NStepAdvantage(Advantage):
    r"""
    Implementation of n-step advantage given by the equation:

    .. math:: \hat{A}_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})
     - V_{t+n-1}(S_{t+1})

    """

    def __init__(self, gamma, n=None):
        self.n_step_reward = NStepReward(gamma, n)

    def discounted(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> np.ndarray:
        return self.n_step_reward.discounted(rewards, values, dones) - values[:, :-1]
