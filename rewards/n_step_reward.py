from rewards.reward import Reward
from rewards.utils import discount
import numpy as np


class NStepReward(Reward):
    r"""
    Implementation of n-step reward given by the equation:

    .. math:: G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})

    """

    def __init__(self, gamma: float, n: int = None):
        """
        :param gamma: discount value. Should be between `(0, 1]`
        :param n: (optional) number of steps to compute reward over. If `None` then calculates it till
               the end of episode
        """
        self.gamma = gamma
        self.n = n

    def discounted(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> np.ndarray:
        if self.n is None:
            return discount(rewards, values[:, -1], dones, self.gamma)
        discounted = np.zeros_like(rewards)
        for start in range(rewards.shape[1]):
            end = min(start + self.n, rewards.shape[1])
            discounted[:, start] = discount(rewards[:, start:end], values[:, end], dones[:, start:end],
                                            self.gamma)[:, 0]
        return discounted
