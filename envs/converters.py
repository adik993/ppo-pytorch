from abc import abstractmethod, ABCMeta
from typing import Tuple

import numpy as np
from gym import Space
import gym.spaces as spaces
from torch import Tensor
from torch.distributions import Categorical, Distribution, Normal


class Converter(metaclass=ABCMeta):
    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, ...]:
        """
        Returns a tuple of integers representing the shape of the observation to be passed as input to the
        model

        :return: tuple of integers representing the shape of the observation/
        """
        raise NotImplementedError('Implement me')

    @property
    @abstractmethod
    def output_shape(self) -> Tuple[int, ...]:
        """
        Returns a tuple of integers representing the shape of the action to be returned from the model

        :return: tuple of integers representing the shape of the action
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

    def action(self, tensor: Tensor) -> np.ndarray:
        """
        Converts logits to action

        :param tensor: logits(output from the model before calling activation function) parametrizing action space
                       distribution
        :return: a tensor containing the action
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

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self.space.n,

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self.space.n,

    def distribution(self, logits: Tensor) -> Distribution:
        return Categorical(logits=logits)

    def reshape_as_input(self, array: np.ndarray, recurrent: bool):
        """
        Basically converts array to one-hot encoded vectors

        :param array: array of shape ``N*T`` with discrete values
        :param recurrent: if true returned array is of shape ``(N,)``
        :return: if ``recurrent`` is ``True`` array of shape ``N*T*``(value returned from ``shape`` property) else:
                 ``N*``(value returned from ``shape`` property)
        """
        out = np.zeros((array.size, self.input_shape[0]), dtype=np.uint8)
        out[np.arange(array.size), array.ravel()] = 1
        out.shape = array.shape + (self.input_shape[0],)
        return out if recurrent else out.reshape(array.shape[0] * array.shape[1], -1)

    def action(self, tensor: Tensor) -> np.ndarray:
        return self.distribution(tensor).sample().cpu().detach().numpy()


class BoxConverter(Converter):
    """
    Utility class to handle ``gym.spaces.Box`` observation/action space
    """

    def __init__(self, space: spaces.Box) -> None:
        self.space = space

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self.space.shape

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return tuple([i * 2 for i in self.space.shape])

    def distribution(self, logits: Tensor) -> Distribution:
        assert logits.size(1) % 2 == 0
        mid = logits.size(1) // 2
        loc = logits[:, :mid]
        scale = logits[:, mid:].exp()
        return Normal(loc, scale)

    def reshape_as_input(self, array: np.ndarray, recurrent: bool):
        return array if recurrent else array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])

    def action(self, tensor: Tensor) -> Tensor:
        return self.distribution(logits=tensor).sample().cpu().detach().numpy().clip(self.space.low, self.space.high)


if __name__ == '__main__':
    import torch

    assert isinstance(Converter.for_space(spaces.Discrete(2)), DiscreteConverter)
    assert isinstance(Converter.for_space(spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)), BoxConverter)

    states = np.array([
        [0, 1, 2],
        [2, 0, 1]
    ])
    converter = DiscreteConverter(spaces.Discrete(3))
    assert converter.input_shape == (3,)
    assert converter.output_shape == (3,)
    assert np.alltrue(converter.reshape_as_input(states, True) == np.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    ]))
    assert np.alltrue(converter.reshape_as_input(states, False) == np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ]))
    distribution = converter.distribution(torch.tensor([[0.1, 0.7, 0.2],
                                                        [0.9, 0.05, 0.05]]))
    assert isinstance(distribution, Categorical)
    assert distribution.sample().shape == (2,)

    states = np.array([
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        [[0.7, 0.8], [0.9, 0.99], [1.0, 1.0]]
    ])
    converter = BoxConverter(spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32))
    assert converter.input_shape == (2,)
    assert converter.output_shape == (4,)
    assert np.allclose(converter.reshape_as_input(states, True), states)
    assert np.allclose(converter.reshape_as_input(states, False), np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
        [0.9, 0.99],
        [1.0, 1.0]
    ]))
    distribution = converter.distribution(torch.tensor([[0., 0., 1., 1.],
                                                        [1., 2., 0., 5.]]))
    assert isinstance(distribution, Normal)
    assert distribution.sample().shape == (2, 2)
