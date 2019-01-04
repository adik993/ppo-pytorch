import numpy as np

from normalizers.normalizer import Normalizer


class NoNormalizer(Normalizer):
    """
    Does no normalization on the array. Handy for observation spaces like ``gym.Discrete``
    """
    def partial_fit(self, array: np.ndarray) -> None:
        pass

    def transform(self, array: np.ndarray) -> np.ndarray:
        return array
