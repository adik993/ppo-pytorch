import numpy as np

from gym import Space

from agents.agent import Agent


class RandomAgent(Agent):
    """
    Just a random agent. It acts randomly by sampling action space
    """

    # noinspection PyMissingConstructor
    def __init__(self, action_space: Space):
        self.action_space = action_space

    def act(self, state) -> np.ndarray:
        return np.asarray([self.action_space.sample() for _ in state])

    def _train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        pass
