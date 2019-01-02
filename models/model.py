from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
from torch import nn, Tensor
from torch.utils.data import Dataset


class Model(nn.Module, metaclass=ABCMeta):
    """
    Base class for all ``nn.Module`` to be used in the ``Agent``'s. It provides basic interface to get the value
    for a given state or a policy logits. Additionally implementations of this class are also responsible of choosing
    the way dataset is sliced by implementing ``dataset`` function that returns ``Dataset`` for example ``LSTM``
    implementations must return ``Dataset`` that returns sequences in each call to get item.

    For ``LSTM`` implementations the model should also heep track of the internal state of the ``LSTM`` cells.

    ``factory`` method is here jus to help keep static typing clean when constructing ``Agent`` objects and should
    return a object that will construct desired implementation of the ``Model``
    """

    def __init__(self, state_space: Tuple[int, ...], action_space: Tuple[int, ...]):
        """
        All the implementations must have this constructor signature
        :param state_space: tuple with state space size
        :param action_space: tuple with action space size
        """
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space

    @property
    @abstractmethod
    def recurrent(self) -> bool:
        """
        :return: whether the model is recurrent or not
        """
        raise NotImplementedError('Implement me')

    @abstractmethod
    def value(self, states: Tensor) -> Tensor:
        """
        Given a state returns predicted value(total reward till the end of the episode that can be collected from this
        state onwards)
        :param states: array of shape that is returned by ``Converter.reshape_as_input``
        :return: `tensor of shape ``B*1``
        """
        raise NotImplementedError('Implement me')

    @abstractmethod
    def policy_logits(self, states: Tensor) -> Tensor:
        """
        Given a state returns logits containing information about actions to be taken. Logits should be passed to
        ``Converter.action`` which will run them through appropriate ``torch.distributions.Distribution`` and return
        the action that then can be passed back to ``MultiEnv``
        :param states: array of shape that is returned by ``Converter.reshape_as_input``
        :return: tensor of shape ``B*action_space``
        """
        raise NotImplementedError('Implement me')

    @abstractmethod
    def dataset(self, *arrays: np.ndarray) -> Dataset:
        """
        Returns ``torch.utils.data.Dataset`` that will return training examples valid for this model
        :param arrays: arrays to be transformed into dataset eg. ``model.dataset(states, actions, rewards)``
        :return: a dataset
        """
        raise NotImplementedError('Implement me')

    @staticmethod
    @abstractmethod
    def factory() -> 'ModelFactory':
        """
        It's here to keep static typing clean when constructing ``Agent`` objects
        :return: factory object that will instantiate the ``Model`` implementation
        """
        raise NotImplementedError('Implement me')


class ModelFactory:
    """
    Constructs ``Model`` objects
    """

    @abstractmethod
    def create(self, state_space: Tuple, action_space: Tuple) -> Model:
        """
        Instantiates ``Model`` object
        :param state_space: state space to be passed to model constructor
        :param action_space: action space to be passed to model constructor
        :return:
        """
        raise NotImplementedError('Implement me')
