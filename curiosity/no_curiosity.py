from typing import List, Generator

import numpy as np
import torch
from torch import Tensor, nn

from curiosity.base import Curiosity, CuriosityFactory
from envs import Converter


class NoCuriosity(Curiosity):
    """
    Placeholder class to be used when agent does not need curiosity. For example in environments that has dense reward.
    """

    # noinspection PyMissingConstructor
    def __init__(self):
        pass

    def reward(self, rewards: np.ndarray, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return rewards

    def loss(self, policy_loss: Tensor, states: Tensor, next_states: Tensor, actions: Tensor) -> Tensor:
        return policy_loss

    def parameters(self) -> Generator[nn.Parameter, None, None]:
        yield from ()

    @staticmethod
    def factory() -> CuriosityFactory:
        return NoCuriosityFactory()


class NoCuriosityFactory(CuriosityFactory):
    def create(self, state_converter: Converter, action_converter: Converter):
        return NoCuriosity()
