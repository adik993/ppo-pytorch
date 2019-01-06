import numpy as np
from torch import nn, Tensor
from torch.utils.data import Dataset

from envs import Converter
from models.datasets import NonSequentialDataset
from models.model import Model, ModelFactory


class MLP(Model):
    """
    MLP model that is able to handle one dimensional state and action spaces. It does not care abut time series.
    So each example returned by the ``Dataset`` returned from ``dataset` method is a tuple of
    a single eg. state, action and reward

    This MLP model is a actor critic style one with shared input layers for both policy and value.
    """

    def __init__(self, state_space: Converter, action_space: Converter):
        assert len(state_space.shape) == 1, 'Only flat spaces supported by MLP model'
        assert len(action_space.shape) == 1, 'Only flat action spaces supported by MLP model'
        super().__init__(state_space, action_space)
        self.input = nn.Sequential(
            nn.Linear(state_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.policy_out = self.action_space.policy_out_model(64)
        self.value_out = nn.Linear(64, 1)

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
    def create(self, state_space: Converter, action_space: Converter) -> Model:
        return MLP(state_space, action_space)
