from collections import Counter

import torch.nn as nn
from abc import ABCMeta, abstractmethod

from torch import Tensor


class Reporter(metaclass=ABCMeta):
    def __init__(self, report_interval: int = 1):
        self.counter = Counter()
        self.graph_initialized = False
        self.report_interval = report_interval
        self.t = 0

    def will_report(self, tag: str) -> bool:
        return self.counter[tag] % (self.report_interval + 1) == 0

    def scalar(self, tag: str, value: float):
        if self.will_report(tag):
            self._scalar(tag, value, self.counter[tag])
        self.counter[tag] += 1

    def graph(self, model: nn.Module, input_to_model: Tensor):
        if not self.graph_initialized:
            self._graph(model, input_to_model)
            self.graph_initialized = True

    @abstractmethod
    def _scalar(self, tag: str, value: float, step: int):
        raise NotImplementedError('Implement me')

    def _graph(self, model: nn.Module, input_to_model: Tensor):
        raise NotImplementedError('Implement me')
