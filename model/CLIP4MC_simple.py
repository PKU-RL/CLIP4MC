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
from module import CrossEn, AllGather

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


class CLIP4MC_simple(nn.Module):
    def __init__(self,
                 frame_num: int,
                 share_sequence_encoder: bool,
                 share_adapter: bool,
                 use_brief_text: bool,
                 use_action: bool,
                 pretrained_clip=None,
                 ):
        """
        Args:
            frame_num:  number of frames in a video clip
            pretrained_clip: pretrained clip model
        """

        super().__init__()
        self.vit = build_ViT(pretrained_clip)
        self.gpt = build_GPT(pretrained_clip)

        self.text_flow = [[self.gpt]]
        self.video_flow = [[self.vit]]

        self.temporal_encoder = build_sequence_encoder('temporal_encoder_config')
        self.video_flow.append([self.temporal_encoder])
        if use_brief_text:
            self.brief_text_encoder = build_sequence_encoder('brief_text_encoder_config')
            self.text_flow.append([self.brief_text_encoder])

        if share_sequence_encoder:
            self.difference_encoder = self.temporal_encoder
            self.temporal_difference_encoder = self.temporal_encoder
            if use_action:
                if use_brief_text:
                    self.action_encoder = self.brief_text_encoder
                else:
                    self.action_encoder = build_sequence_encoder('action_encoder_config')
                    self.text_flow.append([self.action_encoder])
        else:
            self.difference_encoder = build_sequence_encoder('difference_encoder_config')
            self.temporal_difference_encoder = build_sequence_encoder('temporal_difference_encoder_config')
            self.video_flow[-1].append(self.difference_encoder)
            self.video_flow[-1].append(self.temporal_difference_encoder)
            if use_action:
                self.action_encoder = build_sequence_encoder('action_encoder_config')
                if use_brief_text:
                    self.text_flow[-1].append(self.action_encoder)
                else:
                    self.text_flow.append([self.action_encoder])

        self.video_adapter = build_adapter('video_adapter_config')
        self.text_adapter = build_adapter('text_adapter_config')
        self.video_flow.append([self.video_adapter])
        self.text_flow.append([self.text_adapter])
        if share_adapter:
            self.motion_adapter = self.video_adapter
            if use_action:
                self.action_adapter = self.text_adapter
        else:
            self.motion_adapter = build_adapter('motion_adapter_config')
            self.video_flow[-1].append(self.motion_adapter)
            if use_action:
                self.action_adapter = build_adapter('action_adapter_config')
                self.text_flow[-1].append(self.action_adapter)

        self.logit_scale = build_logit_scale(pretrained_clip)

        self.video_layer_num = [max([module.layers for module in modules]) for modules in self.video_flow]
        self.text_layer_num = [max([module.layers for module in modules]) for modules in self.text_flow]

        self.video_layers = sum(self.video_layer_num)
        self.text_layers = sum(self.text_layer_num)
        self.cross_layers = 1
        self.layers = self.cross_layers + max(self.video_layers, self.text_layers)

        self.frame_num = frame_num
        self.use_action = use_action
        self.use_brief_text = use_brief_text
        
        self.video_loss_fct = CrossEn()
        self.motion_loss_fct = CrossEn()

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
        if motion_frame_embedding is None:
            motion_frame_embedding = frame_embedding
        B, T, D = motion_frame_embedding.shape
        video_embedding = self.temporal_encoder(frame_embedding)
        # (batch*frames-1, 2, embed_dim)
        consequence_embedding = torch.stack((motion_frame_embedding[:, 1:], motion_frame_embedding[:, :-1]),
                                            dim=2).reshape(
            B * (T - 1), 2, -1)
        # (batch, frames-1, embed_dim)
        difference_embedding = self.difference_encoder(consequence_embedding).reshape(B, T - 1, -1)
        motion_embedding = self.temporal_difference_encoder(difference_embedding)  # (batch, embed_dim)

        video_embedding = self.video_adapter(video_embedding)  # (batch, embed_dim)
        motion_embedding = self.motion_adapter(motion_embedding)  # (batch, embed_dim)
        motion_embedding = 2 * torch.sigmoid(
            torch.layer_norm(motion_embedding, motion_embedding.shape[1:])) - 1  # (batch, embed_dim)

        video_embedding = video_embedding / video_embedding.norm(dim=-1, keepdim=True)
        motion_embedding = motion_embedding / motion_embedding.norm(dim=-1, keepdim=True)

        return video_embedding, motion_embedding

    def get_text_embedding(self, text, entity_mask=None, action_mask=None):
        if self.use_action or self.use_brief_text:
            action_mask = action_mask.bool()
            entity_mask = entity_mask.bool()
            text_mask = entity_mask | action_mask

        text_embedding = self.gpt.get_hidden_state(text, full=True)

        if self.use_action:
            action_embedding, action_embedding_mask = select_embedding(text_embedding, action_mask,
                                                                       self.action_encoder.max_seq_len)
            action_embedding = self.action_encoder(action_embedding, action_embedding_mask)
            action_embedding = self.action_adapter(action_embedding)
            action_embedding = action_embedding / action_embedding.norm(dim=-1, keepdim=True)

        if self.use_brief_text:
            brief_text_embedding, brief_text_embedding_mask = select_embedding(text_embedding, text_mask,
                                                                               self.brief_text_encoder.max_seq_len)
            text_embedding = self.brief_text_encoder(brief_text_embedding, brief_text_embedding_mask)
        else:
            text_embedding = text_embedding[torch.arange(text_embedding.shape[0]), text.argmax(dim=-1)]

        text_embedding = self.text_adapter(text_embedding)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        return (text_embedding, action_embedding) if self.use_action else text_embedding

    def get_logits(self, video_features, text_features, eval=False):
        video_embedding, motion_embedding = video_features
        if self.use_action:
            text_embedding, action_embedding = text_features
            logit1 = self.logit_scale.exp() * video_embedding @ text_embedding.t()
            logit2 = self.logit_scale.exp() * motion_embedding @ action_embedding.t()
        else:
            text_embedding = text_features
            logit1 = self.logit_scale.exp() * video_embedding @ text_embedding.t()
            logit2 = self.logit_scale.exp() * motion_embedding @ text_embedding.t()
        return (logit1, logit2) if not eval else (logit1 + logit2)

    def forward(self, video, text, entity_mask=None, action_mask=None, motion_input=None, train=False, all_gather=True):
        # video: (batch, frames, channels, height, width)
        # text: (batch, tokens)

        frame_embedding = self.get_image_embedding(video)  # (batch, frames, embed_dim)
        
        if motion_input is not None:
            motion_frame_embedding = self.get_image_embedding(motion_input)  # (batch, frames, embed_dim)
            video_embedding, motion_embedding = self.get_video_embedding(frame_embedding, motion_frame_embedding)
        else:
            video_embedding, motion_embedding = self.get_video_embedding(frame_embedding)
        # (batch, embed_dim)

        if self.use_action:
            text_embedding, action_embedding = self.get_text_embedding(text, entity_mask, action_mask)
        else:
            text_embedding = self.get_text_embedding(text, entity_mask, action_mask)

        if all_gather:
            video_embedding = allgather(video_embedding)
            motion_embedding = allgather(motion_embedding)
            text_embedding = allgather(text_embedding)
            if self.use_action:
                action_embedding = allgather(action_embedding)

        if train:
            v2t_matrix = video_embedding @ text_embedding.t()
            v2t_matrix = self.logit_scale.exp() * v2t_matrix
            t2v_matrix = v2t_matrix.t()
            if self.use_action:
                m2t_matrix = motion_embedding @ action_embedding.t()
            else:
                m2t_matrix = motion_embedding @ text_embedding.t()
            m2t_matrix = self.logit_scale.exp() * m2t_matrix
            t2m_matrix = m2t_matrix.transpose(0, -1)

            loss = (self.video_loss_fct(v2t_matrix) + self.video_loss_fct(t2v_matrix) +
                    self.motion_loss_fct(m2t_matrix) + self.motion_loss_fct(t2m_matrix)) / 4
            return loss
        else:
            video_features = [video_embedding, motion_embedding]
            text_features = text_embedding if not self.use_action else [text_embedding, action_embedding]
            return video_features, text_features

    @torch.no_grad()
    def clamp_logit_scale(self, value=100):
        """
        Follow OpenAI CLIP paper's trick to prevent training instability (sec 2.5)
        """
        self.logit_scale.data.clamp_(-np.log(value), np.log(value))