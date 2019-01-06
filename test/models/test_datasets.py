import pytest
import numpy as np
from torch.utils.data import DataLoader
from models.datasets import NonSequentialDataset


@pytest.fixture
def dataset():
    states = np.array([
        [[1, 1], [2, 2], [3, 3], [4, 4]],
        [[5, 5], [6, 6], [7, 7], [8, 8]]
    ])
    rewards = np.array([
        [10, 20, 30, 40],
        [50, 60, 70, 80]
    ])
    dones = np.array([
        [0, 1, 0, 1],
        [0, 1, 0, 1]
    ])
    return NonSequentialDataset(states, rewards, dones)


def test_dataset_length_is_equal_to_flattened_first_two_dimensions(dataset):
    assert len(dataset) == 8


def test_examples_dimensions_and_values_are_valid(dataset):
    for i in range(len(dataset)):
        state, reward, done = dataset[i]
        assert np.alltrue(state == np.array([i + 1, i + 1]))
        assert reward == (i + 1) * 10
        assert done == i % 2


def test_works_with_dataloader(dataset):
    for state, reward, done in DataLoader(dataset, batch_size=2, shuffle=False):
        assert state.shape == (2, 2)
        assert reward.shape == (2,)
        assert done.shape == (2,)
