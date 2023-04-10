from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle
from functools import lru_cache
from typing import *

import numpy as np
import torch
import yaml
from torchvision import transforms

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)['data_config']

MC_IMAGE_SIZE = config['image_size']
MC_IMAGE_MEAN = config['image_mean']
MC_IMAGE_STD = config['image_std']
CLIP_FRAME_NUM = config['clip_frame']


def torch_normalize(tensor: torch.Tensor, mean=MC_IMAGE_MEAN, std=MC_IMAGE_STD, inplace=False):
    """
    Adapted from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#normalize

    Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """

    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor, dtype=torch.float32)
    if not inplace:
        tensor = tensor.clone()

    assert tensor.dtype == torch.float32, "Only support float32"
    tensor.div_(255.0)

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            f"std evaluated to zero after conversion to {dtype}, leading to division by zero."
        )
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.sub_(mean).div_(std)
    return tensor


image_transform = transforms.Compose([
    transforms.RandomResizedCrop(MC_IMAGE_SIZE, scale=(0.2, 1.), interpolation=transforms.InterpolationMode.BICUBIC),
])


@lru_cache(None)
def get_processed_list(dataset_log_file: str, dataset: Literal['train', 'test'] = None):
    import json
    with open(dataset_log_file, 'r') as f:
        datasets = json.load(f)
    if dataset is None:
        return datasets
    assert dataset in datasets, f'No such dataset {dataset}.'
    return datasets[dataset]


def get_processed_len(dataset_log_file, dataset: Literal["train", "test"] = None):
    return len(get_processed_list(dataset_log_file, dataset))


def load_processed_data(dataset_log_file, data_id: int, use_mask: bool = True,
                        dataset: Literal["train", "test"] = None):
    processed_list = get_processed_list(dataset_log_file, dataset)

    assert data_id < len(processed_list), \
        f'Index {data_id} is beyond the length of processed {dataset} dataset, {len(processed_list)}. '

    file_path = processed_list[data_id]
    with open(os.path.join(file_path, 'text_input.pkl'), 'rb') as f:
        text_input = pickle.load(f)
    with open(os.path.join(file_path, 'video_input.pkl'), 'rb') as f:
        video_input = pickle.load(f)
    T = len(video_input)
    gap = T // CLIP_FRAME_NUM
    video_input = np.array(video_input[::gap])
    return (video_input, *text_input) if use_mask else (video_input, text_input)
