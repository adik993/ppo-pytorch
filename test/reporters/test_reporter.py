import pytest
from torch import nn as nn, Tensor

from reporters import Reporter


class MockReporter(Reporter):
    def __init__(self, report_interval: int = 1):
        super().__init__(report_interval)

    def _scalar(self, tag: str, value: float, step: int):
        pass

    def _graph(self, model: nn.Module, input_to_model: Tensor):
        pass


@pytest.fixture
def reporter(mocker):
    reporter = MockReporter(report_interval=3)
    mocker.spy(reporter, '_scalar')
    mocker.spy(reporter, '_graph')
    return reporter


def test_scalar_respects_report_interval(reporter: Reporter):
    reporter.scalar('a', 1)
    reporter.scalar('a', 2)
    reporter.scalar('b', 1)
    reporter.scalar('a', 3)
    reporter.scalar('a', 4)
    reporter.scalar('a', 5)
    assert reporter._scalar.call_count == 3
    assert [args for args, kwargs in reporter._scalar.call_args_list] == [('a', 1, 0), ('b', 1, 0), ('a', 5, 4)]


def test_will_report_returns_true_when_interval_criteria_is_met(reporter: Reporter):
    assert reporter.will_report('a')
    reporter.scalar('a', 1)
    assert not reporter.will_report('a')


def test_graph_is_only_called_once(reporter: Reporter):
    reporter.graph(None, None)
    reporter.graph(None, None)
    reporter.graph(None, None)
    reporter.graph(None, None)
    assert reporter._graph.call_count == 1
