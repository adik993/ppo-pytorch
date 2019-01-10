from torch import nn as nn, Tensor

from reporters import Reporter


class LogReporter(Reporter):
    def __init__(self, filters: set = None, report_interval: int = 1):
        super().__init__(report_interval)
        self.filter = filters if filters is not None else set()

    def will_report(self, tag: str):
        return tag in self.filter and super().will_report(tag)

    def _scalar(self, tag: str, value: float, step: int):
        print(f'[{tag}] {value}')

    def _graph(self, model: nn.Module, input_to_model: Tensor):
        pass
