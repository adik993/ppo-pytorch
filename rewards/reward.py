import numpy as np


class Reward:
    """
    Base interface for all reward functions like n-step reward
    """

    def discounted(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """
        Returns discounted reward

        Legend for dimensions:
         * ``N`` - number of parallel agents
         * ``T`` - number of time steps

        :param rewards: array of shape `N*T` containing rewards for each time step
        :param values: array of shape `N*(T+1)` containing value estimates returned by the agent for all time step
        :param dones: array of shape `N*T` containing information about episodes ends
        :return: array of shape `N*T` with discounted reward for each time step
        """
        raise NotImplementedError('Implement me')
