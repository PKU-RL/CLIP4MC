from typing import *
import torch
from torch.utils.data import DataLoader

from .dataset import NaiveDataset


def get_naive_dataloader(dataset_log_file: str,
                         use_mask: bool,
                         batch_size: int,
                         dataset: Literal["train", "val", "test"] = None,
                         num_workers: int = 8):
    """
    Get a naive dataloader of processed data.

    Args:
        process_name: data process type.
        batch_size: batch size.
        dataset: which dataset to use, train, val or test.
        num_workers: number of workers.

    Returns:
        A naive dataloader of processed data.
    """
    naive_dataset = NaiveDataset(dataset_log_file,use_mask,dataset)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        naive_sampler = torch.utils.data.distributed.DistributedSampler(naive_dataset, shuffle=(dataset == 'train'))
        n_devices = torch.distributed.get_world_size()
        batch_size = batch_size // n_devices
    else:
        naive_sampler = None

    naive_dataloader = DataLoader(naive_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=False,
                                  shuffle=(naive_sampler is None),
                                  sampler=naive_sampler,
                                  drop_last=True)
    return naive_dataloader, naive_sampler, len(naive_dataset)
