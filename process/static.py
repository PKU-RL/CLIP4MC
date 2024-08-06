import os
import pickle
import json
from typing import Literal
from functools import lru_cache
from math import ceil
import torch
import numpy as np
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)['data_config']

# constant used in processing
CLIP_WORD_LENGTH = config['clip_word_length']
CLIP_WORD_STRIDE = config['clip_word_stride']

TOKENIZER = config['tokenizer']
TOKENIZE_MAX_LENGTH = config['tokenize_max_length']
HIDDEN_STATE_MAX_LENGTH = config['hidden_state_max_length']

SUMMARY_MODEL = config['summary_model']
SUMMARY_MAX_LENGTH = config['summary_max_length']
SUMMARY_MIN_LENGTH = config['summary_min_length']

MC_IMAGE_SIZE = config['image_size']
MC_IMAGE_RESIZE = (MC_IMAGE_SIZE[1], MC_IMAGE_SIZE[0])
MC_IMAGE_MEAN = config['image_mean']
MC_IMAGE_STD = config['image_std']

CLIP_FRAME_NUM = config['clip_frame_num']
VIDEO_CLIP_LENGTH = config['video_clip_length']

TEST_RATIO = config['test_ratio']
VAL_RATIO = config['val_ratio']

# data directories
ORIGIN_DATA_DIR = config['origin_data_dir']
PROCESSED_DATA_DIR = config['processed_data_dir']

video_dir = os.path.join(ORIGIN_DATA_DIR, 'video')
video_frame_dir = os.path.join(ORIGIN_DATA_DIR, 'video_frame')
transcript_dir = os.path.join(ORIGIN_DATA_DIR, 'transcripts')
keyword_file = os.path.join(ORIGIN_DATA_DIR, 'keywords.json')


# origin data file paths/ids
def get_origin_list(data_type: Literal["video", "transcript"], return_ids: bool = True):
    if data_type == "video":
        dirt = video_dir
        ext = '.mp4'
    elif data_type == "transcript":
        dirt = transcript_dir
        ext = '.srt'
    else:
        raise NotImplementedError

    return [f.split('.')[0] for f in os.listdir(dirt) if f.endswith(ext)] if return_ids else \
        [os.path.join(dirt, f) for f in os.listdir(dirt) if f.endswith(ext)]


# processed data file paths (collected by hands of workers)
def split_dataset(process_name: str):
    processed_dir = os.path.join(PROCESSED_DATA_DIR, process_name)
    assert os.path.isdir(processed_dir), f'processed data {process_name} does not exist.'
    ext = '.pkl'
    workers = [n for n in os.listdir(processed_dir) if n.startswith('worker_')]

    def get_list(worker_list):
        data_list = []
        for worker in worker_list:
            data_list.extend([os.path.join(processed_dir, worker, f)
                              for f in os.listdir(os.path.join(processed_dir, worker)) if f.endswith(ext)])
        return data_list

    test_split = ceil(len(workers) * TEST_RATIO)
    test_workers, train_workers = workers[:test_split], workers[test_split:]
    test_data_list = get_list(test_workers)
    train_data_list = get_list(train_workers)

    import random
    random.shuffle(train_data_list)
    val_split = ceil(len(train_data_list) * VAL_RATIO)
    val_data_list, train_data_list = train_data_list[:val_split], train_data_list[val_split:]

    dataset_info_path = os.path.join(processed_dir, 'dataset_info.json')
    with open(dataset_info_path, 'w') as f:
        json.dump({'test': test_data_list, 'train': train_data_list, 'val': val_data_list}, f)
    return len(test_data_list), len(train_data_list), len(val_data_list)


@lru_cache(None)
def get_processed_list(dataset: Literal['train', 'val', 'test'] = None):

    dataset_info_path = "/json/file/of/dataset"
    with open(dataset_info_path, 'r') as f:
        datasets = json.load(f)
    if dataset is None:
        return datasets
    assert dataset in datasets, f'No such dataset {dataset}.'
    return datasets[dataset]


def get_processed_len(dataset: Literal["train", "val", "test"] = None):
    return len(get_processed_list(dataset))


def load_processed_data(data_id: int, use_mask: bool = True, dataset: Literal["train", "val", "test"] = None):
    processed_list = get_processed_list(dataset)

    assert data_id < len(processed_list), \
        f'Index {data_id} is beyond the length of processed {dataset} dataset, {len(processed_list)}. '
    
    file_path = processed_list[data_id]

    with open(os.path.join(file_path,'text_input.pkl'),'rb') as f:
       text_input = pickle.load(f)['tokens']

    with open(os.path.join(file_path,'video_input.pkl'), 'rb') as f:
        video_input_chosen = pickle.load(f)

    video_input_chosen = np.array(video_input_chosen).transpose(0,3,1,2)
    
    length = video_input_chosen.shape[0]
    k = length//16
    video_input = video_input_chosen[::k, ::-1,:,:]

    return (text_input,video_input) if use_mask else (text_input,video_input)


def load_processed_data_new(data_id: int, use_mask: bool = True, dataset: Literal["train", "val", "test"] = None):
    processed_list = get_processed_list(dataset)

    assert data_id < len(processed_list), \
        f'Index {data_id} is beyond the length of processed {dataset} dataset, {len(processed_list)}. '

    file_path = processed_list[data_id]
    with open(os.path.join(file_path, 'text_input.pkl'), 'rb') as f:
        text_input = pickle.load(f)["tokens"]

    with open(os.path.join(file_path, 'video_input.pkl'), 'rb') as f:
        video_input_chosen = pickle.load(f)
    video_input_chosen = np.array(video_input_chosen).transpose(0,3,1,2)

    length = video_input_chosen.shape[0]
    k = length // 16
    video_input = video_input_chosen[::k, ::-1, :, :]
    with open(os.path.join(file_path,'size.json'), "r") as f:
       size = json.load(f)

    size = torch.tensor(size)

    return (text_input,0,0,size,video_input) if use_mask else (text_input,size,video_input)



# Requirements: Data from different workers must be collected from different videos.
def save_processed_data(process_name: str, worker_id: int, data_id: int, data):
    processed_dir = os.path.join(PROCESSED_DATA_DIR, process_name)
    worker_dir = os.path.join(processed_dir, f'worker_{worker_id}')
    if not os.path.exists(worker_dir):
        os.makedirs(worker_dir)
    file_path = os.path.join(worker_dir, f'{data_id}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
