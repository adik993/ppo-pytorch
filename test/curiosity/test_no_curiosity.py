import numpy as np
import pytest
import torch

from curiosity import NoCuriosity


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
def rewards():
    return np.array([
        [1., 1., ],
        [-1., 2., ]
    ])


@pytest.fixture
def curiosity():
    return NoCuriosity.factory().create(None, None)


def test_reward_returned_unaltered(curiosity, states, actions, rewards):
    np.testing.assert_equal(curiosity.reward(rewards, states, actions), rewards)


def test_loss_returned_unaltered(curiosity, states, actions):
    states = torch.tensor(states)
    actions = torch.tensor(actions)
    policy_loss = torch.tensor(1.2)
    np.testing.assert_equal(curiosity.loss(policy_loss, states, states, actions), policy_loss)
