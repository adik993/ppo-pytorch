import pytest

from reporters import Reporter
from reporters.log_reporter import LogReporter


@pytest.fixture
def reporter(mocker):
    reporter = LogReporter({'allowed1', 'allowed2'}, report_interval=2)
    mocker.spy(reporter, '_scalar')
    return reporter


def test_reporter_respects_filters(reporter: Reporter):
    reporter.scalar('not_allowed', 1)
    reporter.scalar('allowed1', 0)
    reporter.scalar('allowed2', 0)
    assert reporter._scalar.call_count == 2
    assert [args for args, _ in reporter._scalar.call_args_list] == [('allowed1', 0, 0), ('allowed2', 0, 0)]


def test_scalar_respects_report_interval(reporter: Reporter):
    reporter.scalar('allowed1', 1)
    reporter.scalar('allowed1', 2)
    reporter.scalar('allowed2', 1)
    reporter.scalar('allowed1', 3)
    reporter.scalar('allowed1', 4)
    reporter.scalar('allowed1', 5)
    assert reporter._scalar.call_count == 3
    assert [args for args, kwargs in reporter._scalar.call_args_list] == [('allowed1', 1, 0), ('allowed2', 1, 0),
                                                                          ('allowed1', 4, 3)]


def test_will_report_returns_true_when_interval_criteria_is_met(reporter: Reporter):
    assert not reporter.will_report('notallowed')
    assert reporter.will_report('allowed1')
    reporter.scalar('allowed1', 1)
    assert not reporter.will_report('allowed1')
