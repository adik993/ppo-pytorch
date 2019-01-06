import pytest
import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from torch.distributions import Categorical, Normal

from envs.converters import DiscreteConverter, BoxConverter, Converter, NormalDistributionModule
from normalizers import NoNormalizer, StandardNormalizer


@pytest.mark.parametrize("space,expected_converter_type", [
    (spaces.Discrete(2), DiscreteConverter),
    (spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32), BoxConverter)
])
def test_converter_for_space_returns_appropriate_converter(space, expected_converter_type):
    assert isinstance(Converter.for_space(space), expected_converter_type)


class TestDiscreteConverter:
    @pytest.fixture
    def converter(self):
        return DiscreteConverter(spaces.Discrete(3))

    @pytest.fixture
    def states(self):
        return np.array([
            [0, 1, 2],
            [2, 0, 1]
        ])

    @pytest.fixture
    def policy_logits(self):
        return torch.tensor([[0.1, 0.7, 0.2],
                             [0.9, 0.05, 0.05]])

    def test_discrete_property_returns_true(self, converter):
        assert converter.discrete

    def test_shape_extracted_from_space(self, converter):
        assert converter.shape == (3,)

    def test_distance_is_cross_entropy_loss(self, converter):
        np.testing.assert_allclose(converter.distance(torch.tensor([[0.1, 0.9, 0.1]]), torch.tensor([1.])).item(),
                                   0.641,
                                   atol=1e-3)

    def test_reshape_as_input_for_recurrent_leaves_array_as_n_agents_by_n_steps_array(self, converter, states):
        np.testing.assert_equal(converter.reshape_as_input(states, True), states)

    def test_reshape_as_input_for_non_recurrent_flattens_n_agents_and_timesteps_dimension(self, converter, states):
        np.testing.assert_equal(converter.reshape_as_input(states, False), np.array([[0], [1], [2], [2], [0], [1]]))

    def test_distribution_is_categorical(self, converter, policy_logits):
        distribution = converter.distribution(policy_logits)
        assert isinstance(distribution, Categorical)

    def test_distribution_shape_is_valid(self, converter, policy_logits):
        distribution = converter.distribution(policy_logits)
        assert distribution.sample().shape == (2,)

    def test_no_normalizer_returned(self, converter):
        assert isinstance(converter.state_normalizer(), NoNormalizer)

    def test_policy_out_model_is_linear(self, converter):
        assert isinstance(converter.policy_out_model(1), nn.Linear)


class TestBoxConverter:
    @pytest.fixture
    def converter(self):
        return BoxConverter(spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32))

    @pytest.fixture
    def states(self):
        return np.array([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[0.7, 0.8], [0.9, 0.99], [1.0, 1.0]]
        ])

    @pytest.fixture
    def policy_logits(self):
        return torch.tensor([[0., 0., 1., 1.],
                             [1., 2., 0., 5.]])

    def test_discrete_property_returns_false(self, converter):
        assert not converter.discrete

    def test_shape_extracted_from_space(self, converter):
        assert converter.shape == (2,)

    def test_distance_is_mse_and_accepts_logits_as_first_param(self, converter):
        np.testing.assert_allclose(converter.distance(torch.tensor([[.5, 1e-3]]), torch.tensor([[.5]])), 0.0, atol=1e-3)

    def test_reshape_as_input_for_recurrent_leaves_array_as_n_agents_by_n_steps_array(self, converter, states):
        np.testing.assert_equal(converter.reshape_as_input(states, True), states)

    def test_reshape_as_input_for_non_recurrent_flattens_n_agents_and_timesteps_dimension(self, converter, states):
        np.testing.assert_equal(converter.reshape_as_input(states, False), np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
            [0.9, 0.99],
            [1.0, 1.0]
        ]))

    def test_distribution_is_normal(self, converter, policy_logits):
        distribution = converter.distribution(policy_logits)
        assert isinstance(distribution, Normal)

    def test_distribution_shape_is_valid(self, converter, policy_logits):
        distribution = converter.distribution(policy_logits)
        assert distribution.sample().shape == (2, 2)

    def test_standard_normalizer_returned(self, converter):
        assert isinstance(converter.state_normalizer(), StandardNormalizer)

    def test_policy_out_model_is_normal_distribution_module(self, converter):
        assert isinstance(converter.policy_out_model(1), NormalDistributionModule)
