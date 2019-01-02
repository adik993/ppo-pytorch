import numpy as np

from torch.utils.data import Dataset, DataLoader


class NonSequentialDataset(Dataset):
    """
     * ``N`` - number of parallel environments
     * ``T`` - number of time steps explored in environments

    Dataset that flattens ``N*T*...`` arrays into ``B*...`` (where ``B`` is equal to ``N*T``) and returns such rows
    one by one. So basically we loose information about sequence order and we return
    for example one state, action and reward per row.

    It can be used for ``Model``'s that does not need to keep the order of events like MLP models.

    For ``LSTM`` use another implementation that will slice the dataset differently
    """

    def __init__(self, *arrays: np.ndarray) -> None:
        """
        :param arrays: arrays to be flattened from ``N*T*...`` to ``B*...`` and returned in each call to get item
        """
        super().__init__()
        self.arrays = [array.reshape(-1, *array.shape[2:]) for array in arrays]

    def __getitem__(self, index):
        return [array[index] for array in self.arrays]

    def __len__(self):
        return len(self.arrays[0])


if __name__ == '__main__':
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

    dataset = NonSequentialDataset(states, rewards, dones)
    assert len(dataset) == 8
    for i in range(len(dataset)):
        state, reward, done = dataset[i]
        assert np.alltrue(state == np.array([i + 1, i + 1]))
        assert reward == (i + 1) * 10
        assert done == i % 2

    for state, reward, done in DataLoader(dataset, batch_size=2, shuffle=False):
        print(state)
        print(reward)
        print(done)
