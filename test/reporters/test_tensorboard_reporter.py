from reporters import TensorBoardReporter


def test_summary_writer_constructed_with_valid_logdir(mocker):
    SummaryWriter = mocker.patch('reporters.tensorboard_reporter.SummaryWriter')
    TensorBoardReporter('a')
    SummaryWriter.assert_called_with('a', '')


def test_scalar_calls_writer_add_scalar(mocker):
    SummaryWriter = mocker.patch('reporters.tensorboard_reporter.SummaryWriter')
    reporter = TensorBoardReporter()
    reporter.scalar('a', 1)
    SummaryWriter.return_value.add_scalar.assert_called_with('a', 1, 0)


def test_graph_calls_writer_add_graph(mocker):
    SummaryWriter = mocker.patch('reporters.tensorboard_reporter.SummaryWriter')
    reporter = TensorBoardReporter()
    reporter.graph('module', 'input')
    SummaryWriter.return_value.add_graph.assert_called_with('module', 'input')


def test_scalar_respects_report_interval(mocker):
    SummaryWriter = mocker.patch('reporters.tensorboard_reporter.SummaryWriter')
    reporter = TensorBoardReporter(report_interval=3)
    reporter.scalar('a', 1)
    reporter.scalar('a', 2)
    reporter.scalar('b', 1)
    reporter.scalar('a', 3)
    reporter.scalar('a', 4)
    reporter.scalar('a', 5)
    assert SummaryWriter.return_value.add_scalar.call_count == 3
    assert [args for args, kwargs in SummaryWriter.return_value.add_scalar.call_args_list] == [('a', 1, 0), ('b', 1, 0),
                                                                                               ('a', 5, 4)]

def test_graph_is_only_called_once(mocker):
    SummaryWriter = mocker.patch('reporters.tensorboard_reporter.SummaryWriter')
    reporter = TensorBoardReporter()
    reporter.graph(None, None)
    reporter.graph(None, None)
    reporter.graph(None, None)
    reporter.graph(None, None)
    assert SummaryWriter.return_value.add_graph.call_count == 1


def test_will_report_returns_true_when_interval_criteria_is_met(mocker):
    mocker.patch('reporters.tensorboard_reporter.SummaryWriter')
    reporter = TensorBoardReporter(report_interval=3)
    assert reporter.will_report('a')
    reporter.scalar('a', 1)
    assert not reporter.will_report('a')
