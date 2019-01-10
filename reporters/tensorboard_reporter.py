from tensorboardX import SummaryWriter
from torch import nn as nn, Tensor

from reporters.reporter import Reporter


class TensorBoardReporter(Reporter):
    def __init__(self, logdir: str = None, comment: str = '', report_interval: int = 1):
        super().__init__(report_interval)
        self.writer = SummaryWriter(logdir, comment)

    def _scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def _graph(self, model: nn.Module, input_to_model: Tensor):
        self.writer.add_graph(model, input_to_model)
