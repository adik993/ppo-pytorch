import pytest

import numpy as np

from normalizers import StandardNormalizer


@pytest.fixture
def normalizer():
    return StandardNormalizer()


def test_partial_fit_updates_mean_and_std_incrementally(normalizer):
    normalizer.partial_fit(np.array([[1., 2.]]))
    normalizer.partial_fit(np.array([[2., 3.]]))
    np.testing.assert_allclose(normalizer.transform(np.array([[1., 2.], [2., 3.]])), np.array([[-1.414214, 0.],
                                                                                               [0., 1.414214]]),
                               atol=1e-2)


def test_partial_fit_transform_first_fits_then_transforms(normalizer):
    actual = normalizer.partial_fit_transform(np.array([[1., 2.], [2., 3.]]))
    np.testing.assert_allclose(actual, np.array([[-1.414214, 0.],
                                                 [0., 1.414214]]), atol=1e-2)
