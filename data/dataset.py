from typing import *
import random
import torch
import numpy as np
# import h5py
from torch.utils.data import Dataset

from process import get_processed_len, load_processed_data, load_processed_data_new
from process import torch_normalize, image_transform


class NaiveDataset(Dataset):
    def __init__(self, use_mask: bool=True, dataset: Literal["train", "val", "test"] = None):
        """
        Build a naive dataset of processed data.

        Args:
            process_name: data process type.
            dataset: which dataset to use, train, val or test.
        """
        self.dataset = dataset
        self.len = get_processed_len(dataset)
        self.use_mask = use_mask
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        data = list(load_processed_data(idx, self.use_mask, self.dataset))
        video = data[-1]

        video = torch.as_tensor(video.copy())

        if self.dataset == 'train':

            video = image_transform(torch.as_tensor(video))

            video = torch_normalize(video.float())
        else:
            video = torch_normalize(torch.as_tensor(video).float())
        data[-1] = video

        return data


class NewDataset(Dataset):
    def __init__(self, use_mask: bool = True, dataset: Literal["train", "val", "test"] = None):
        """
        Build a naive dataset of processed data.

        Args:
            process_name: data process type.
            dataset: which dataset to use, train, val or test.
        """
        self.dataset = dataset
        self.len = get_processed_len(dataset)
        self.use_mask = use_mask

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        data = list(load_processed_data_new(idx, self.use_mask, self.dataset))
        video = data[-1]



        video = torch.as_tensor(video.copy())

        if self.dataset == 'train':

            video = image_transform(torch.as_tensor(video))

            video = torch_normalize(video.float())
        else:
            video = torch_normalize(torch.as_tensor(video).float())
        data[-1] = video

        return data
