from typing import Iterable, List, Tuple, Union

import numpy as np
from envs import Runner, MultiEnv, Converter
from models import ModelFactory
import torch


class Agent:
    """
    Base interface for agents
    """

    def __init__(self, env: MultiEnv, model_factory: ModelFactory) -> None:
        self.env = env
        self.state_converter = Converter.for_space(self.env.observation_space)
        self.action_converter = Converter.for_space(self.env.action_space)
        self.model = model_factory.create(self.state_converter.input_shape, self.action_converter.output_shape)
        self.device: torch.device = None
        self.dtype: torch.dtype = None
        self.numpy_dtype: object = None

    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Acts in the environment. Returns the action for the given state

        Note: ``N`` in the dimensions stands for number of parallel environments being explored

        :param state: state of shape N * (state space shape) that we want to know the action for
        :return: the action which is array of shape N * (action space shape)
        """
        reshaped_states = self.state_converter.reshape_as_input(state[:, None, :], self.model.recurrent)
        logits = self.model.policy_logits(torch.tensor(reshaped_states, device=self.device))
        return self.action_converter.action(logits)

    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        """
        Trains the agent using previous experience.
        Legend for the dimensions of input arrays:
         * ``N`` - number of parallel environments being explored(see ``MultiEnv``)
         * ``T`` - number of time steps that were run on the environment

        :param states: of shape N * T * (state space shape)
        :param actions: of shape N * T * (action space shape)
        :param rewards: of shape N * T * 1
        :param dones: of shape N * T * 1
        """
        raise NotImplementedError('Implement me')

    def learn(self, epochs: int, n_steps: int, render: bool = False):
        """
        Trains the agent for ``epochs`` number of times by running simulation on the environment for ``n_steps``

        :param epochs: number of epochs of training
        :param n_steps: number of steps made in the environment each epoch
        :param render: whether to render the environment during learning
        """
        for epoch in range(epochs):
            states, actions, rewards, dones = Runner(self.env, self).run(n_steps, render)
            self.train(states, actions, rewards, dones)
            print(f'Epoch: {epoch} done')

    def to(self, device: torch.device, dtype: torch.dtype, numpy_dtype: Union[object, str]) -> None:
        """
        Transfers the agent's model to device
        :param device: device to transfer agent to
        :param dtype: dtype to which cast the model parameters
        :param numpy_dtype: dtype to use for the environment. *Must* be the same as ``dtype`` parameter
        :return:
        """
        self.device = device
        self.dtype = dtype
        self.numpy_dtype = numpy_dtype
        self.model.to(device, dtype)
        self.env.astype(numpy_dtype)

    def _tensors_to_device(self, *tensors: torch.Tensor) -> List[torch.Tensor]:
        return [tensor.to(self.device, self.dtype) for tensor in tensors]

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, device=self.device, dtype=self.dtype)
