from torch import nn as nn, Tensor

from reporters.reporter import Reporter


class NoReporter(Reporter):

    def will_report(self, tag: str) -> bool:
        return False

    def scalar(self, tag: str, value: float):
        pass

    def graph(self, model: nn.Module, input_to_model: Tensor):
        pass

    def _scalar(self, tag: str, value: float, step: int):
        pass

    def _graph(self, model: nn.Module, input_to_model: Tensor):
        pass
