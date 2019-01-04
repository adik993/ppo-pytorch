from abc import ABCMeta, abstractmethod

import numpy as np


class Normalizer(metaclass=ABCMeta):
    """
    Base class for normalizers used to normalize the sate/reward during training before inputing it to the model
    or a curiosity module
    """

    @abstractmethod
    def partial_fit(self, array: np.ndarray) -> None:
        """
        Incrementally fit the normalizer eg. incrementally calculate mean
        :param array: array of shape ``N*T`` or ``N*T*any`` to calculate statistics on
        """
        raise NotImplementedError('Implement me')

    @abstractmethod
    def transform(self, array: np.ndarray) -> np.ndarray:
        """
        Normalizes the array using insights gathered with ``partial_fit``
        :param array: array of shape ``N*T`` or ``N*T*any`` to be normalized
        :return:  normalized array of same shape as input
        """
        raise NotImplementedError('Implement me')

    def partial_fit_transform(self, array: np.ndarray) -> np.ndarray:
        """
        Handy method to run ``partial_fit`` and ``transform`` at once
        :param array: array of shape ``N*T`` or ``N*T*any`` to be normalized
        :return: normalized array of same shape as input
        """
        self.partial_fit(array)
        return self.transform(array)
