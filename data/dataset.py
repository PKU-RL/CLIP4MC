from typing import *
import random
import torch
import numpy as np
from torch.utils.data import Dataset

from .process import get_processed_len, load_processed_data
from .process import torch_normalize, image_transform


class NaiveDataset(Dataset):
    def __init__(self, dataset_log_file: str, use_mask: bool=True, dataset: Literal["train", "val", "test"] = None):
        """
        Build a naive dataset of processed data.

        Args:
            process_name: data process type.
            dataset: which dataset to use, train, val or test.
        """
        self.dataset = dataset
        self.dataset_log_file = dataset_log_file
        self.len = get_processed_len(dataset_log_file, dataset)
        self.use_mask = use_mask

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = list(load_processed_data(self.dataset_log_file, idx, self.use_mask, self.dataset))
        video = data[0]
        video = torch.as_tensor(video)
        video = video.permute(0, 3, 1, 2)

        if self.dataset == 'train':            
            video = image_transform(video)

        video = torch_normalize(video.float())
        data[0] = video
        return data
