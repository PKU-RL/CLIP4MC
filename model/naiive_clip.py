from __future__ import annotations
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from typing import *

from module import build_GPT, build_ViT, build_logit_scale, build_adapter, build_sequence_encoder, CrossEn, \
                    AllGather

allgather = AllGather.apply


class NaiveCLIP(nn.Module):
    def __init__(self, frame_num: int, pretrained_clip=None):
        """
        Args:
            frame_num:  number of frames in a video clip
            pretrained_clip: pretrained clip model
        """
        super().__init__()
        self.vit = build_ViT(pretrained_clip)
        self.gpt = build_GPT(pretrained_clip)
        self.video_adapter = build_adapter('video_adapter_config')
        self.text_adapter = build_adapter('text_adapter_config')
        self.temporal_encoder = build_sequence_encoder('temporal_encoder_config')
        self.logit_scale = build_logit_scale(pretrained_clip)

        self.video_layers = self.vit.layers + self.temporal_encoder.layers + self.video_adapter.layers
        self.text_layers = self.gpt.layers + self.text_adapter.layers
        self.cross_layers = 1
        self.layers = self.cross_layers + max(self.video_layers, self.text_layers)
        self.clip_frame_num = frame_num

        self.loss_fct = CrossEn()

    def get_layer(self, layer: int, layer_type: Literal['video', 'text', 'cross']):
        if layer_type == 'video':
            if layer < self.vit.layers:
                return self.vit.get_layer(layer)
            elif layer < self.vit.layers + self.temporal_encoder.layers:
                return self.temporal_encoder.get_layer(layer - self.vit.layers)
            elif layer < self.video_layers:
                return self.video_adapter.get_layer(layer - self.vit.layers - self.temporal_encoder.layers)
        elif layer_type == 'text':
            if layer < self.gpt.layers:
                return self.gpt.get_layer(layer)
            elif layer < self.text_layers:
                return self.text_adapter.get_layer(layer - self.gpt.layers)
        elif layer_type == 'cross':
            if layer == 0:
                return self.logit_scale,
        return []
    
    def get_image_embedding(self, image):
        return self.vit(image)
    
    def get_video_embedding(self, frame_embedding):
        video_embedding = self.temporal_encoder(frame_embedding)
        video_embedding = self.video_adapter(video_embedding)
        video_embedding = video_embedding / video_embedding.norm(dim=-1, keepdim=True)
        return video_embedding

    def get_text_embedding(self,text,entity_mask=None,action_mask=None):
        text_embedding = self.gpt(text)
        text_embedding = self.text_adapter(text_embedding)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding
    
    def get_logits(self,video_features,text_features,eval=False):
        return self.logit_scale.exp() * video_features @ text_features.t()
    
    def forward(self, video, text, train=False, all_gather=True):
        # video: (batch, frames, channels, height, width)
        # text: (batch, tokens)
        frame_embedding = self.get_image_embedding(video)  # (batch, frames, embed_dim)
        clips_features = self.get_video_embedding(frame_embedding)  # (batch, embed_dim)
        text_features = self.get_text_embedding(text)

        if all_gather:
            text_features = allgather(text_features)  # (Batch, embed_dim)
            clips_features = allgather(clips_features)  # (Batch, embed_dim)

        if train:
            v2t_matrix = self.get_logits(clips_features, text_features)  # (Batch, Batch)
            t2v_matrix = v2t_matrix.transpose(0, 1)  # (Batch,  Batch)
            loss = (self.loss_fct(v2t_matrix) + self.loss_fct(t2v_matrix)) / 2
            return loss
        else:
            return clips_features, text_features

    @torch.no_grad()
    def clamp_logit_scale(self, value=100):
        """
        Follow OpenAI CLIP paper's trick to prevent training instability (sec 2.5)
        """
        self.logit_scale.data.clamp_(-np.log(value), np.log(value))
