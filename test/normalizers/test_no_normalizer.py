import pytest

import numpy as np

from normalizers import NoNormalizer


@pytest.fixture
def normalizer():
    return NoNormalizer()


def test_partial_fit_does_nothing(normalizer):
    normalizer.partial_fit_transform(np.array([[1., 2.], [2., 3.]]))
    np.testing.assert_equal(normalizer.transform(np.array([[1., 2.], [2., 3.]])), np.array([[1., 2.], [2., 3.]]))


def test_transform_returns_input_array(normalizer):
    np.testing.assert_equal(normalizer.transform(np.array([[1., 2.], [2., 3.]])), np.array([[1., 2.], [2., 3.]]))


def test_partial_fit_transform_resurns_input_array(normalizer):
    np.testing.assert_equal(normalizer.partial_fit_transform(np.array([[1., 2.], [2., 3.]])), np.array([[1., 2.],
                                                                                                        [2., 3.]]))
