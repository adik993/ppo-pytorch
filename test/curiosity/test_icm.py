import numpy as np
import pytest
import torch
from gym.spaces import Box, Discrete

from curiosity import ICM, MlpICMModel
from envs import Converter


@pytest.fixture
def states():
    return np.array([
        [[1., 1.], [2., 2.], [3., 3.]],
        [[4., 4.], [4., 4.], [5., 5.]]
    ])


@pytest.fixture
def actions():
    return np.array([
        [0., 1., ],
        [1., 0., ]
    ])


@pytest.fixture
def icm():
    curiosity = ICM.factory(MlpICMModel.factory(), 1, 2, 0.5, 1.).create(
        Converter.for_space(Box(0, 5, (2,), np.float32)),
        Converter.for_space(Discrete(2))
    )
    curiosity.to(torch.device('cpu'), torch.float32)
    return curiosity


def test_reward_has_valid_shape(icm, states, actions):
    rewards = icm.reward(np.zeros_like(actions), states, actions)
    assert rewards.shape == (2, 2)


def test_loss_is_scalar(icm, states, actions):
    loss = icm.loss(torch.tensor(10., dtype=torch.float32),
                    torch.tensor(states[:, :-1].reshape(-1, 2), dtype=torch.float32),
                    torch.tensor(states[:, 1:].reshape(-1, 2), dtype=torch.float32),
                    torch.tensor(actions.reshape(-1), dtype=torch.float32))
    assert loss.shape == ()
