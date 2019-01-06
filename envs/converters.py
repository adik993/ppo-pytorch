from abc import abstractmethod, ABCMeta
from typing import Tuple

import numpy as np
from gym import Space
import gym.spaces as spaces
import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Distribution, Normal
from torch.nn import CrossEntropyLoss, MSELoss

from normalizers import Normalizer, StandardNormalizer, NoNormalizer


class Converter(metaclass=ABCMeta):
    @property
    @abstractmethod
    def discrete(self) -> bool:
        """
        Whether underlying space is discrete or not
        :return: ``True`` if space is discrete aka. ``gym.spaces.Discrete``, ``False`` otherwise
        """
        raise NotImplementedError('Implement me')

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """
        Returns a tuple of integers representing the shape of the observation to be passed as input to the
        model

        :return: tuple of integers representing the shape of the observation/
        """
        raise NotImplementedError('Implement me')

    @abstractmethod
    def distribution(self, logits: Tensor) -> Distribution:
        """
        Returns a distribution appropriate for a ``gym.Space`` parametrized using provided ``logits``

        :return: logits returned by the model
        """
        raise NotImplementedError('Implement me')

    @abstractmethod
    def reshape_as_input(self, array: np.ndarray, recurrent: bool):
        """
        Converts the array to match the shape returned by the ``shape`` property

        :param array: array of shape ``N*T*(any shape produced by the underlying ``gym.Space``
        :param recurrent: whether reshaping fo recurrent model or not
        :return: array of shape ``N*T*``(shape returned by ``shape`` property)
        """
        raise NotImplementedError('Implement me')

    @abstractmethod
    def action(self, tensor: Tensor) -> Tensor:
        """
        Converts logits to action

        :param tensor: logits(output from the model before calling activation function) parametrizing action space
                       distribution
        :return: a tensor containing the action
        """
        raise NotImplementedError('Implement me')

    @abstractmethod
    def distance(self, policy_logits: Tensor, y: Tensor) -> Tensor:
        """
        Returns the distance between two tensors of an underlying space
        :param policy_logits: predictions
        :param y: actual values
        :return: distance/loss
        """
        raise NotImplementedError('Implement me')

    @abstractmethod
    def state_normalizer(self) -> Normalizer:
        """
        Returns the normalizer to be used for the observation
        :return: normalizer instance
        """
        raise NotImplementedError('Implement me')

    @abstractmethod
    def policy_out_model(self, in_features: int) -> nn.Module:
        """
        Returns the output layer for the policy that is appropriate for a given action space
        :return: torch module that accepts ``in_features`` and outputs values for policy
        """
        raise NotImplementedError('Implement me')

    @staticmethod
    def for_space(space: Space):
        if isinstance(space, spaces.Discrete):
            return DiscreteConverter(space)
        elif isinstance(space, spaces.Box):
            return BoxConverter(space)


class DiscreteConverter(Converter):
    """
    Utility class to handle ``gym.spaces.Discrete`` observation/action space
    """

    def __init__(self, space: spaces.Discrete) -> None:
        self.space = space
        self.loss = CrossEntropyLoss()

    @property
    def discrete(self) -> bool:
        return True

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.space.n,

    def distribution(self, logits: Tensor) -> Distribution:
        return Categorical(logits=logits)

    def reshape_as_input(self, array: np.ndarray, recurrent: bool):
        return array if recurrent else array.reshape(array.shape[0] * array.shape[1], -1)

    def action(self, tensor: Tensor) -> Tensor:
        return self.distribution(tensor).sample()

    def distance(self, policy_logits: Tensor, y: Tensor) -> Tensor:
        return self.loss(policy_logits, y.long())

    def state_normalizer(self) -> Normalizer:
        return NoNormalizer()

    def policy_out_model(self, in_features: int) -> nn.Module:
        return nn.Linear(in_features, self.shape[0])


class BoxConverter(Converter):
    """
    Utility class to handle ``gym.spaces.Box`` observation/action space
    """

    def __init__(self, space: spaces.Box) -> None:
        self.space = space
        self.loss = MSELoss()

    @property
    def discrete(self) -> bool:
        return False

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.space.shape

    def distribution(self, logits: Tensor) -> Distribution:
        assert logits.size(1) % 2 == 0
        mid = logits.size(1) // 2
        loc = logits[:, :mid]
        scale = logits[:, mid:]
        return Normal(loc, scale)

    def reshape_as_input(self, array: np.ndarray, recurrent: bool):
        return array if recurrent else array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])

    def action(self, tensor: Tensor) -> Tensor:
        min = torch.tensor(self.space.low, device=tensor.device)
        max = torch.tensor(self.space.high, device=tensor.device)
        return torch.max(torch.min(self.distribution(logits=tensor).sample(), max), min)

    def distance(self, policy_logits: Tensor, y: Tensor) -> Tensor:
        return self.loss(self.action(policy_logits), y)

    def state_normalizer(self) -> Normalizer:
        return StandardNormalizer()

    def policy_out_model(self, in_features: int) -> nn.Module:
        return NormalDistributionModule(in_features, self.shape[0])


class NormalDistributionModule(nn.Module):
    def __init__(self, in_features: int, n_action_values: int):
        super().__init__()
        self.policy_mean = nn.Linear(in_features, n_action_values)
        self.policy_std = nn.Parameter(torch.zeros(1, n_action_values))

    def forward(self, x):
        policy = self.policy_mean(x)
        policy_std = self.policy_std.expand_as(policy).exp()
        return torch.cat((policy, policy_std), dim=-1)
