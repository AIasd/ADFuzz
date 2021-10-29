from itertools import repeat

import torch


def _repeater(dataloader):
    for loader in repeat(dataloader):
        for data in loader:
            yield data


def _dataloader(data, sampler, batch_size, num_workers):
    return torch.utils.data.DataLoader(
            data, batch_size=batch_size, num_workers=num_workers,
            sampler=sampler, drop_last=True, pin_memory=True)


def infinite_dataloader(data, sampler, batch_size, num_workers):
    return _repeater(_dataloader(data, sampler, batch_size, num_workers))


class Wrap(object):
    def __init__(self, data, sampler, batch_size, samples, num_workers):
        self.data = infinite_dataloader(data, sampler, batch_size, num_workers)
        self.samples = samples

    def __iter__(self):
        for i in range(self.samples):
            yield next(self.data)

    def __len__(self):
        return self.samples
