import numpy as np

from rewards.utils import discount


def test_discount_returns_discounted_reward_and_respects_episode_ends():
    # given
    rewards = np.array([
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1]
    ], np.float32)
    dones = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ], np.float32)
    estimate_of_last = np.array([6, 0], np.float32)
    lam = 0.9
    # when
    actual = discount(rewards, estimate_of_last, dones, lam)
    # then
    expected = np.array([
        [rewards[0, 0] + lam * (rewards[0, 1] + lam * rewards[0, 2]),
         rewards[0, 1] + lam * rewards[0, 2],
         rewards[0, 2],
         rewards[0, 3] + lam * (rewards[0, 4] + lam * estimate_of_last[0]),
         rewards[0, 4] + lam * estimate_of_last[0],
         ],
        [rewards[1, 0] + lam * (rewards[1, 1] + lam * (
                rewards[1, 2] + lam * (rewards[1, 3] + lam * (rewards[1, 4] + lam * estimate_of_last[1])))),
         rewards[1, 1] + lam * (
                 rewards[1, 2] + lam * (rewards[1, 3] + lam * (rewards[1, 4] + lam * estimate_of_last[1]))),
         rewards[1, 2] + lam * (rewards[1, 3] + lam * (rewards[1, 4] + lam * estimate_of_last[1])),
         rewards[1, 3] + lam * (rewards[1, 4] + lam * estimate_of_last[1]),
         rewards[1, 4] + lam * estimate_of_last[1]
         ]
    ])
    np.testing.assert_allclose(actual, expected)
