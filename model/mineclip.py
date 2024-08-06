from __future__ import annotations
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from typing import *

from module import build_GPT, build_ViT, build_logit_scale, build_adapter, build_sequence_encoder

from module import CrossEn, AllGather, VisionTransformer, GPT, AdapterHead, SequenceTransformer

allgather = AllGather.apply


def select_embedding(embedding, mask, seq_len):
    ans_embedding = []
    ans_embedding_mask = []
    for msk, emb in zip(mask, embedding):
        select_emb = torch.masked_select(emb, msk.unsqueeze(-1)).view(-1, emb.size(-1))
        tmp_emb = torch.zeros(seq_len, emb.size(-1), device=emb.device)
        tmp_msk = torch.zeros(seq_len, device=emb.device, dtype=torch.bool)
        select_emb = select_emb[:seq_len]
        tmp_emb[:select_emb.size(0)] = select_emb
        tmp_msk[:select_emb.size(0)] = 1
        ans_embedding.append(tmp_emb)
        ans_embedding_mask.append(tmp_msk)
    return torch.stack(ans_embedding, dim=0), torch.stack(ans_embedding_mask, dim=0)


class NaiveCLIP(nn.Module):
    def __init__(self,
                 frame_num: int,
                 use_brief_text: bool,
                 use_action: bool,
                 pretrained_clip=None):
        """
        Args:
            frame_num:  number of frames in a video clip
            pretrained_clip: pretrained clip model
        """

        super().__init__()
        self.vit = build_ViT(pretrained_clip)
        self.gpt = build_GPT(pretrained_clip)
        self.sigmoid = torch.nn.Sigmoid()
        self.text_flow = [[self.gpt]]
        self.video_flow = [[self.vit]]

        self.temporal_encoder = build_sequence_encoder('temporal_encoder_config')

        self.video_flow.append([self.temporal_encoder])

        self.video_adapter = build_adapter('video_adapter_config')
        self.text_adapter = build_adapter('text_adapter_config')
        self.video_flow.append([self.video_adapter])
        self.text_flow.append([self.text_adapter])


        self.logit_scale = build_logit_scale()

        self.video_layer_num = [max([module.layers for module in modules]) for modules in self.video_flow]
        self.text_layer_num = [max([module.layers for module in modules]) for modules in self.text_flow]

        self.video_layers = sum(self.video_layer_num)
        self.text_layers = sum(self.text_layer_num)
        self.cross_layers = 1
        self.layers = self.cross_layers + max(self.video_layers, self.text_layers)

        self.frame_num = frame_num
        self.use_action = use_action
        self.use_brief_text = use_brief_text

        self.loss_fct = CrossEn()

    def get_layer(self, layer: int, layer_type: Literal['video', 'text', 'cross']):
        if layer_type == 'video':
            for i, l in enumerate(self.video_layer_num):
                if layer < l:
                    ans = []
                    for module in self.video_flow[i]:
                        if layer < module.layers:
                            ans += module.get_layer(layer)
                    return ans
                layer -= l
        elif layer_type == 'text':
            for i, l in enumerate(self.text_layer_num):
                if layer < l:
                    ans = []
                    for module in self.text_flow[i]:
                        if layer < module.layers:
                            ans += module.get_layer(layer)
                    return ans
                layer -= l
        elif layer_type == 'cross':
            if layer == 0:
                return self.logit_scale,
        return []

    def get_image_embedding(self, image):
        return self.vit(image)

    def get_video_embedding(self, frame_embedding, motion_frame_embedding=None):

        B, T, D = frame_embedding.shape
        video_embedding = self.temporal_encoder(frame_embedding)
        video_embedding = self.video_adapter(video_embedding)  # (batch, embed_dim)
        video_embedding = video_embedding / video_embedding.norm(dim=-1, keepdim=True)

        return video_embedding


    def get_text_embedding(self, text, entity_mask, action_mask):
        action_mask = action_mask.bool()
        entity_mask = entity_mask.bool()
        text_mask = entity_mask | action_mask

        text_embedding = self.gpt.get_hidden_state(text, full=True)

        text_embedding = text_embedding[torch.arange(text_embedding.shape[0]), text.argmax(dim=-1)]

        text_embedding = self.text_adapter(text_embedding)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        return text_embedding


    def get_logits(self, video_features, text_features):
        video_embedding = video_features


        text_embedding = text_features
        logit1 = self.logit_scale.exp() * video_embedding @ text_embedding.t()
            
        return logit1


    def forward(self, text, video, motion_input=None, train=False, all_gather=True):
        # video: (batch, frames, channels, height, width)
        # text: (batch, tokens)
        B, T, C, H, W = video.shape
        
        frame_embedding = self.get_image_embedding(video)  # (batch, frames, embed_dim)

        video_embedding = self.get_video_embedding(frame_embedding)

        text_embedding = self.get_text_embedding(text, text, text)

        if all_gather:
            video_embedding = allgather(video_embedding)

            text_embedding = allgather(text_embedding)


        if train:

            v2t_matrix = video_embedding @ text_embedding.t()
            v2t_matrix = self.logit_scale.exp() * v2t_matrix
            t2v_matrix = v2t_matrix.t()

            loss = (self.loss_fct(v2t_matrix) + self.loss_fct(t2v_matrix)) / 2

            return loss
        else:
            video_features = [self.logit_scale.exp()*video_embedding]

            text_features = text_embedding 
            
            return video_features, text_features

    @torch.no_grad()
    def clamp_logit_scale(self, value=100):
        """
        Follow OpenAI CLIP paper's trick to prevent training instability (sec 2.5)
        """
        self.logit_scale.data.clamp_(-np.log(value), np.log(value))

