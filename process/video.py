from __future__ import annotations
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os

import cv2
import numpy as np
from math import ceil
import torch
import pickle

from .static import video_dir, video_frame_dir
from .static import MC_IMAGE_RESIZE
from .image import torch_normalize
from functools import lru_cache
from model import NaiveCLIP

def get_origin_video(vid: str):
    """
    get the video capture.

    Args:
        vid (str): Video file process_name.

    Returns:
        cv2.VideoCapture: Video capture.
    """
    video_path = os.path.join(video_dir, vid + '.mp4')
    assert os.path.exists(video_path), f'video {video_path} does not exist.'
    vc = cv2.VideoCapture(video_path)
    assert vc.isOpened(), f'video {video_path} cannot be opened.'
    return vc


def get_video(vid: str):
    """
    Get directory of video frames.

    Args:
        vid (str): Video id.

    """
    path = os.path.join(video_frame_dir, vid)
    assert os.path.exists(path), f'video directory {path} does not exist.'
    return path



def video_to_clips(vid: str,
                   clip_pos: list[int | float],
                   clip_len: int,
                   frame_num: int,
                   cvt_color: bool = True,
                   resize: tuple[int, int] | None = MC_IMAGE_RESIZE,
                   transpose: bool = True):
    """
    get the video clips.

    Args:
        vid (str): Video file process_name.
        clip_pos (list[int | float]): list of clip positions (the second of midpoint) sorted temporally.
        clip_len (int): Clip length (seconds).
        frame_num (int): Total frame number.
        cvt_color (bool, optional): Whether to convert color. Defaults to True.
        resize (tuple[int, int], optional): Resize the video. Defaults to (256, 160), reversed image size.
        transpose (bool, optional): Whether to transpose the video into (T,C,H,W). Defaults to True.

    Returns:
        list[np.ndarray]: Video clips.
    """
    video_path = get_video(vid)
    fps = 5  # TODO: get fps from config
    total_frame = len(os.listdir(video_path))

    # num = len(clip_pos)

    clips_start = [int(max(pos - clip_len / 2, 0) * fps) for pos in clip_pos]
    # assert num < 2 or max([clips_start[i] - clips_start[i + 1] for i in range(num - 1)]) <= 0, \
    #     'clip positions are not sorted temporally.'
    clips_end = [min(st + clip_len * fps, total_frame) for st in clips_start]

    clips = []
    for i, (st, ed) in enumerate(zip(clips_start, clips_end)):
        clip = []
        assert frame_num < 3 * (ed - st), f'frame number {frame_num} is too much larger than clip length {ed - st}.'
        time_pos = np.linspace(st, ed, frame_num, endpoint=False, dtype=int)
        for t in time_pos:
            frame = cv2.imread(os.path.join(video_path, f'{t}.png'))
            assert frame is not None, f'frame {t} does not exist.'
            if cvt_color:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize is not None:
                frame = cv2.resize(frame, resize)
            clip.append(frame)

        if len(clip) == frame_num:
            clip = np.stack(clip, axis=0)
            if transpose:
                clip = clip.transpose((0, 3, 1, 2))
            clips.append(clip)
        else:
            raise ValueError(f'clip {i} in video {vid} has {len(clip)} frames, but {frame_num} frames are required.')
    return clips


def video_seg_to_clip_new(clip: str,
                          clip_pos: int | float,
                          clip_length: int,
                          frame_num: int,
                          cvt_color: bool = True,
                          resize: tuple[int, int] | None = MC_IMAGE_RESIZE,
                          transpose: bool = True,
                          device: str | torch.device = 'cuda'):
    """
    get the video segmentation clips.

    Args:
        clip (str): Video file process_name.
        clip_pos (list[int | float]): list of clip positions (the second of midpoint) sorted temporally.
        clip_length (int): Clip length (seconds).
        frame_num (int): Total frame number.
        cvt_color (bool, optional): Whether to convert color. Defaults to True.
        resize (tuple[int, int], optional): Resize the video. Defaults to (256, 160), reversed image size.
        transpose (bool, optional): Whether to trimage_seganspose the video into (T,C,H,W). Defaults to True.

    Returns:
        list[np.ndarray]: Video clips.
    """
    video_path = os.path.join(clip, 'video_input.pkl')
    with open(video_path, 'rb') as f:
        frames = pickle.load(f)
    fps = 10  # TODO: get fps from config
    clip = image_seg(frames, 3, fps, True, device)
    return clip


def video_to_clips_back(vid: str,
                        clip_pos: list[int | float],
                        clip_len: int,
                        frame_num: int,
                        cvt_color: bool = True,
                        resize: tuple[int, int] | None = MC_IMAGE_RESIZE,
                        transpose: bool = True):
    """
    get the video clips.

    Args:
        vid (str): Video file process_name.
        clip_pos (list[int | float]): list of clip positions (the second of midpoint) sorted temporally.
        clip_len (int): Clip length (seconds).
        frame_num (int): Total frame number.
        cvt_color (bool, optional): Whether to convert color. Defaults to True.
        resize (tuple[int, int], optional): Resize the video. Defaults to (256, 160), reversed image size.
        transpose (bool, optional): Whether to transpose the video into (T,C,H,W). Defaults to True.

    Returns:
        list[np.ndarray]: Video clips.
    """
    vc = get_origin_video(vid)
    fps = vc.get(cv2.CAP_PROP_FPS)
    duration = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    num = len(clip_pos)
    clips_start = [int(max(pos - clip_len / 2, 0) * fps) for pos in clip_pos]

    assert num < 2 or max([clips_start[i] - clips_start[i + 1] for i in range(num - 1)]) <= 0, \
        'clip positions are not sorted temporally.'
    clips_end = [min(st + clip_len * fps, duration) for st in clips_start]

    total_end = min(max(clips_end), duration)
    dense_clips = [[]] * num
    active_st = 0
    active_end = 0

    # Reduce the number of times of reading video while loading the whole video is heavy for memory. Here we maintain
    # a window of active clips at current frame. Warning: read a certain frame by set(cv2.CAP_PROP_POS_FRAMES,
    # frame_id) may result in kernel ERROR in concurrent environment.
    while True:
        pos = vc.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = vc.read()
        assert ret, f'video {vid} failed while reading frame {pos}.'

        if pos >= total_end:
            break
        while active_end < num and pos > clips_start[active_end]:
            active_end += 1
        while active_st < num and pos > clips_end[active_st]:
            active_st += 1
        if active_st < active_end:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if cvt_color else frame
            frame = cv2.resize(frame, dsize=resize, interpolation=cv2.INTER_CUBIC) if resize is not None else frame
            for i in range(active_st, active_end):
                dense_clips[i].append(frame)

    clips = []
    for i, dense_clip in enumerate(dense_clips):
        if len(dense_clip) < frame_num:
            res = frame_num - len(dense_clip)
            assert res < len(dense_clip), f'clip {i} of video {vid} has too few frames.'
            augment = np.linspace(0, len(dense_clip) - 1, res, dtype=int)
            for j in augment:
                dense_clip.insert(j, dense_clip[j])
        elif len(dense_clip) > frame_num:
            select = np.linspace(0, len(dense_clip) - 1, frame_num, dtype=int)
            dense_clip = [dense_clip[i] for i in select]

        dense_clip = np.stack(dense_clip, axis=0)
        dense_clip = np.transpose(dense_clip, (0, 3, 1, 2)) if transpose else dense_clip
        clips.append(dense_clip)
    vc.release()
    return clips


def video_to_frames(vid, fps):
    """
    Extract frames from video.

    Args:
        vid (str): Video file process_name.
        fps (int): Frame rate.
    """
    vc = get_origin_video(vid)
    frame_dir = os.path.join(video_frame_dir, vid)
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    frame_num = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    src_fps = vc.get(cv2.CAP_PROP_FPS)
    num = ceil(frame_num / src_fps * fps)
    new_frame_pos = np.linspace(0, frame_num - 1, num, dtype=int)
    gaps = new_frame_pos[1:] - new_frame_pos[:-1]
    for i in range(num):
        ret, frame = vc.read()
        if not ret:
            break
        frame = cv2.resize(frame, dsize=(256, 160), interpolation=cv2.INTER_CUBIC)  # Todo: resize
        cv2.imwrite(os.path.join(frame_dir, f'{i}.png'), frame)
        if i < num - 1:
            for _ in range(gaps[i] - 1):
                vc.read()

    vc.release()
