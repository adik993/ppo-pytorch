import pytest
import numpy as np
import torch
from gym.spaces import Box, Discrete

from envs import Converter
from models import MLP
from models.datasets import NonSequentialDataset


@pytest.fixture
def model():
    return MLP.factory().create(Converter.for_space(Box(0, 1, (4,), np.float32)),
                                Converter.for_space(Discrete(2)))


def test_model_value_output_shape_is_valid(model):
    assert model.value(torch.tensor([[1., 2., 3., 4.], [4., 5., 6., 7.]])).shape == (2, 1)


def test_model_policy_logits_shae_is_valid(model):
    assert model.policy_logits(torch.tensor([[1., 2., 3., 4.], [4., 5., 6., 7.]])).shape == (2, 2)


def test_returned_dataset_is_non_sequential(model):
    assert isinstance(model.dataset(np.array([[1., 2., 3., 4.], [4., 5., 6., 7.]])), NonSequentialDataset)
