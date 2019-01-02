import numpy as np

from rewards import Reward
from rewards.utils import discount


class GeneralizedRewardEstimation(Reward):
    """
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
        return discount(td_errors, np.zeros_like(values[:, 0]), dones, self.lam * self.gamma) + values[:, :-1]


if __name__ == '__main__':
    # given
    rewards = np.array([
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1]
    ], np.float32)
    dones = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ], np.float32)
    values = np.array([
        [-100, 10, 20, 30, 40, 50],
        [-150, 15, 25, 35, 45, 55]
    ], np.float32)
    lam = 0.9
    gamma = 0.8
    # when
    actual = GeneralizedRewardEstimation(gamma, lam).discounted(rewards, values, dones)
    # then
    expected = np.array([
        [(rewards[0, 0] + gamma * values[0, 1] - values[0, 0]) + gamma * lam * (
                (rewards[0, 1] + gamma * values[0, 2] - values[0, 1]) + gamma * lam * (
                rewards[0, 2] - values[0, 2])),
         (rewards[0, 1] + gamma * values[0, 2] - values[0, 1]) + gamma * lam * (rewards[0, 2] - values[0, 2]),
         (rewards[0, 2] - values[0, 2]),
         (rewards[0, 3] + gamma * values[0, 4] - values[0, 3]) + gamma * lam * (
                 rewards[0, 4] + gamma * values[0, 5] - values[0, 4]),
         (rewards[0, 4] + gamma * values[0, 5] - values[0, 4]),
         ],
        [(rewards[1, 0] + gamma * values[1, 1] - values[1, 0]) + gamma * lam * (
                (rewards[1, 1] + gamma * values[1, 2] - values[1, 1]) + gamma * lam * (
                (rewards[1, 2] + gamma * values[1, 3] - values[1, 2]) + gamma * lam * (
                (rewards[1, 3] + gamma * values[1, 4] - values[1, 3]) + gamma * lam * (
                rewards[1, 4] + gamma * values[1, 5] - values[1, 4])))),
         (rewards[1, 1] + gamma * values[1, 2] - values[1, 1]) + gamma * lam * (
                 (rewards[1, 2] + gamma * values[1, 3] - values[1, 2]) + gamma * lam * (
                 (rewards[1, 3] + gamma * values[1, 4] - values[1, 3]) + gamma * lam * (
                 rewards[1, 4] + gamma * values[1, 5] - values[1, 4]))),
         (rewards[1, 2] + gamma * values[1, 3] - values[1, 2]) + gamma * lam * (
                 (rewards[1, 3] + gamma * values[1, 4] - values[1, 3]) + gamma * lam * (
                 rewards[1, 4] + gamma * values[1, 5] - values[1, 4])),
         (rewards[1, 3] + gamma * values[1, 4] - values[1, 3]) + gamma * lam * (
                 rewards[1, 4] + gamma * values[1, 5] - values[1, 4]),
         (rewards[1, 4] + gamma * values[1, 5] - values[1, 4])
         ]
    ]) + values[:, :-1]
    assert np.allclose(expected, actual)
