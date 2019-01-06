import numpy as np
import gym
import pytest

from envs import MultiEnv


@pytest.fixture
def gym_env():
    return gym.make('CartPole-v1')


@pytest.fixture
def env():
    return MultiEnv('CartPole-v1', 2)


class TestMultiEnv:
    @pytest.fixture(autouse=True)
    def close_env(self, env, gym_env):
        yield
        env.close()
        gym_env.close()

    def test_observation_space_matches_gym_env_one(self, env, gym_env):
        assert env.observation_space == gym_env.observation_space

    def test_action_space_matches_gym_env_one(self, env, gym_env):
        assert env.action_space == gym_env.action_space

    def test_returned_values_has_first_dimension_equal_to_n_of_agents(self, env):
        env.reset()
        for i in range(10):
            env.render()
            action = np.array([env.action_space.sample()] * env.n_envs, dtype=np.int64)
            state, reward, done, aux = env.step(action)
            assert state.shape == (2, 4)
            assert reward.shape == (2,)
            assert done.shape == (2,)
