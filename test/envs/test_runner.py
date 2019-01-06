import pytest

from agents import RandomAgent
from envs import Runner, MultiEnv


@pytest.fixture
def runner():
    env = MultiEnv('CartPole-v1', 2)
    return Runner(env, RandomAgent(env.action_space))


def test_runner_return_shapes_are_n_agents_by_n_steps_by_any_shape(runner):
    states, actions, rewards, dones = runner.run(100, False)
    assert states.shape == (2, 101, 4)
    assert actions.shape == (2, 100)
    assert rewards.shape == (2, 100)
    assert dones.shape == (2, 100)
