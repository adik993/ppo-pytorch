from abc import abstractmethod, ABCMeta
from typing import List, Generator

import numpy as np
import torch
from torch import Tensor, nn

from envs import Converter


class Curiosity(metaclass=ABCMeta):
    """
    Base class for curiosity implementations. Curiosity is the idea of calculating additional intrinsic reward for
    the agent, so that even if the reward is sparse agent can still learn something. Intrinsic reward can be for example
    measure of how surprised the agent is with the new state what encourages exploration of undiscovered states.
    """

    def __init__(self, state_converter: Converter, action_converter):
        self.state_converter = state_converter
        self.action_converter = action_converter
        self.device: torch.device = None
        self.dtype: torch.dtype = None

    @abstractmethod
    def reward(self, rewards: np.ndarray, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Returns the modified rewards with the intrinsic reward incorporated
        :param rewards: extrinsic reward returned from the environment as a array of shape ``N*T``
        :param states: array of shape ``N*(T+1)*(state space)`` containing explored states
        :param actions: array of shape ``N*T*(action space)`` containing the actions
        :return: array of same shape as ``rewards`` parameters with the intrinsic reward incorporated
        """
        raise NotImplementedError('Implement me')

    @abstractmethod
    def loss(self, policy_loss: Tensor, states: Tensor, next_states: Tensor, actions: Tensor) -> Tensor:
        """
        Returns the altered ``policy_loss`` that incorporates the curiosity module loss
        :param policy_loss: tensor with one element that is policy loss
        :param states: tensor of shape ``N*(state space)``(or `` N*T*(state space)`` for recurrent) containing states
        :param next_states: tensor of shape ``N*(state space)``(or `` N*T*(state space)`` for recurrent)
               containing next states
        :param actions: tensor of shape ``N*(action space)`` containing actions taken
        :return: the loss altered with the curiosity module loss
        """
        raise NotImplementedError('Implement me')

    @abstractmethod
    def parameters(self) -> Generator[nn.Parameter, None, None]:
        """
        Parameters of the curiosity module to be trained(if any)
        :return:
        """
        raise NotImplementedError('Implement me')

    def to(self, device: torch.device, dtype: torch.dtype) -> None:
        """
        Transfers module to the device casting to dtype
        :param device: ``torch.device`` to transfer to
        :param dtype: ``torch.dtype`` to cast all the parameters to
        """
        self.device = device
        self.dtype = dtype

    def _to_tensors(self, *arrays: np.ndarray) -> List[torch.Tensor]:
        return [torch.tensor(array, device=self.device, dtype=self.dtype) for array in arrays]


class CuriosityFactory(metaclass=ABCMeta):
    """
    Base class for curiosity module factories
    """

    @abstractmethod
    def create(self, state_converter: Converter, action_converter: Converter):
        """
        Creates curiosity module
        :param state_converter: state converter
        :param action_converter: action converter
        :return:
        """
        raise NotImplementedError('Implement me')
