import os
import sys
from abc import ABC, abstractmethod

from d_model.nn_A0_utils import calc_tensor_memsize
from utility.watch import Watch

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
from tqdm import tqdm
import torch


def get_random_point(H, W):
    # get rate in height and width dimension and get corresponding index
    rate_h = np.random.rand()
    rate_w = np.random.rand()
    idx_H = int(np.round(rate_h * (H - 1)))
    idx_W = int(np.round(rate_w * (W - 1)))
    return rate_h, rate_w, idx_H, idx_W


class AbstractDataset(Dataset, ABC):

    def __init__(self):
        super().__init__()
        self.dataset_partition = ''

    @abstractmethod
    def get_namekeys(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


class CustomDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True, sampler: Sampler = None, num_workers: int = 4):
        self.dataset = dataset
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        self.default_shuffle = shuffle
        self.default_sampler = sampler
        self.num_workers = num_workers

    def get_iterator(self, batch_size: int = None, device: str = None, shuffle: bool = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        shuffle = shuffle if shuffle is not None else self.default_shuffle

        for batch in self:
            if device is not None:
                batch = [item.to(device) for item in batch]
            yield batch


if __name__ == '__main__':
    pass
