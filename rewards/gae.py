import numpy as np

from rewards.advantage import Advantage
from rewards.utils import discount


class GeneralizedAdvantageEstimation(Advantage):
    r"""
    Implementation of Generalized Advantage Estimator given by the equation:

    .. math:: \hat{A}_t = \delta_t + \gamma\lambda\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}

    where

    .. math:: \delta_t = r_t + \gamma V(s_{t+1})-V(s_t)

    """

    def __init__(self, gamma, lam):
        """

        :param gamma: essentially it's the discount factor as we know it from n-step rewards
        :param lam: can be interpreted as the `n` in n-step rewards. Specifically setting it to 0 reduces the equation
               to be single step TD error, while setting it to 1 means there is no horizon so estimate over all steps
        """
        self.gamma = gamma
        self.lam = lam

    def discounted(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> np.ndarray:
        td_errors = rewards + self.gamma * values[:, 1:] * (1. - dones) - values[:, :-1]
        return discount(td_errors, np.zeros_like(values[:, 0]), dones, self.lam * self.gamma)
