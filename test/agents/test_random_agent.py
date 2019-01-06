import numpy as np
import pytest
from gym.spaces import Discrete

from agents import RandomAgent


@pytest.fixture
def agent():
    return RandomAgent(Discrete(3))


def test_act_returns_actions_as_array_with_first_dimension_equal_to_states_shape(agent):
    assert agent.act(np.array([[1., 2.], [3., 4.]])).shape == (2,)
