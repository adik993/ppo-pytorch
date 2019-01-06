import numpy as np

from rewards import GeneralizedAdvantageEstimation


def test_gae_returns_valid_values():
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
    actual = GeneralizedAdvantageEstimation(gamma, lam).discounted(rewards, values, dones)
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
    ])
    np.testing.assert_allclose(expected, actual)
