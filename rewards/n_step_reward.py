from rewards.reward import Reward
from rewards.utils import discount
import numpy as np


class NStepReward(Reward):
    """
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
    n_step = 4
    # when
    actual = NStepReward(lam, n_step).discounted(rewards, values, dones)
    # then
    expected = np.array([
        [rewards[0, 0] + lam * (rewards[0, 1] + lam * rewards[0, 2]),
         rewards[0, 1] + lam * rewards[0, 2],
         rewards[0, 2],
         rewards[0, 3] + lam * (rewards[0, 4] + lam * values[0, 5]),
         rewards[0, 4] + lam * values[0, 5],
         ],
        [rewards[1, 0] + lam * (rewards[1, 1] + lam * (rewards[1, 2] + lam * (rewards[1, 3] + lam * values[1, 4]))),
         rewards[1, 1] + lam * (rewards[1, 2] + lam * (rewards[1, 3] + lam * (rewards[1, 4] + lam * values[1, 5]))),
         rewards[1, 2] + lam * (rewards[1, 3] + lam * (rewards[1, 4] + lam * values[1, 5])),
         rewards[1, 3] + lam * (rewards[1, 4] + lam * values[1, 5]),
         rewards[1, 4] + lam * values[1, 5]
         ]
    ])
    assert np.allclose(expected, actual)
