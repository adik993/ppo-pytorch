from typing import Tuple

import numpy as np
from torch import nn, Tensor
from torch.utils.data import Dataset

from models.datasets import NonSequentialDataset
from models.model import Model, ModelFactory


class MLP(Model):
    """
    MLP model that is able to handle one dimensional state and action spaces. It does not care abut time series.
    So each example returned by the ``Dataset`` returned from ``dataset` method is a tuple of
    a single eg. state, action and reward

    This MLP model is a actor critic style one with shared input layers for both policy and value.
    """

    def __init__(self, state_space: Tuple, action_space: Tuple):
        assert len(state_space) == 1, 'Only flat spaces supported by MLP model'
        assert len(action_space) == 1, 'Only flat action spaces supported by MLP model'
        super().__init__(state_space, action_space)
        self.input = nn.Sequential(
            nn.Linear(state_space[0], 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True)
        )
        self.policy_out = nn.Linear(16, action_space[0])
        self.value_out = nn.Linear(16, 1)

    def forward(self, state):
        x = self.input(state)
        policy = self.policy_out(x)
        value = self.value_out(x)
        return policy, value

    @property
    def recurrent(self) -> bool:
        return False

    def value(self, states: Tensor) -> Tensor:
        _, value = self(states)
        return value

    def policy_logits(self, states: Tensor) -> Tensor:
        policy, _ = self(states)
        return policy

    def dataset(self, *arrays: np.ndarray) -> Dataset:
        return NonSequentialDataset(*arrays)

    @staticmethod
    def factory() -> 'ModelFactory':
        return MLPFactory()


class MLPFactory(ModelFactory):
    def create(self, state_space: Tuple, action_space: Tuple) -> Model:
        return MLP(state_space, action_space)


if __name__ == '__main__':
    import torch

    model = MLP.factory().create((4,), (2,))
    assert isinstance(model, MLP)

    assert model.value(torch.tensor([[1., 2., 3., 4.], [4., 5., 6., 7.]])).shape == (2, 1)
    assert model.policy_logits(torch.tensor([[1., 2., 3., 4.], [4., 5., 6., 7.]])).shape == (2, 2)
    assert isinstance(model.dataset(np.array([[1., 2., 3., 4.], [4., 5., 6., 7.]])), NonSequentialDataset)
