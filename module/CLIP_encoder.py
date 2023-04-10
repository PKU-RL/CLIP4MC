"""
Adapted from OpenAI CLIP implementation: https://github.com/openai/CLIP
"""
from __future__ import annotations

from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat


def interpolate_resize_pos_embed(pos_embed, old_size, new_size):
    """
    NOTE: remove cls token from pos_embed first before passing it here
    Args:
        pos_embed: [seq_len, embed_dim]
        old_size: [h, w], seq_len of pos_embed must be equal to h * w
        new_size: [new_h, new_w]
    """
    old_hw, D = pos_embed.size()
    if isinstance(old_size, int):
        old_size = (old_size, old_size)
    if isinstance(new_size, int):
        new_size = (new_size, new_size)
    assert len(old_size) == 2
    assert len(new_size) == 2
    old_h, old_w = old_size
    assert old_h * old_w == old_hw
    pos_embed = rearrange(pos_embed, "(H W) D -> 1 D H W", H=old_h)
    new_embed = torch.nn.functional.interpolate(
        pos_embed, size=new_size, mode="bicubic", align_corners=False
    )
    new_embed = rearrange(new_embed, "1 D H W -> (H W) D")
    assert new_embed.size() == (new_size[0] * new_size[1], D)
    return new_embed


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            resolution: int | tuple[int, int],
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            output_dim: int,
    ):
        super().__init__()
        self._resolution = resolution
        self._patch_size = patch_size

        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        for r in resolution:
            assert (r % patch_size) == 0, f"{resolution} is not divisible by {patch_size}"

        self._layers = layers
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width ** -0.5
        self.cls_token = nn.Parameter(scale * torch.randn(width))
        self.pos_embed = nn.Parameter(
            scale * torch.randn((resolution[0] // patch_size) * (resolution[1] // patch_size) + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, output_dim))
        self.layers = layers + 2

    def get_layer(self, layer: int):
        assert 0 <= layer < self.layers, f"layer {layer} is out of range"
        if layer == 0:
            return self.conv1, self.cls_token, self.pos_embed, self.ln_pre
        elif layer == self.layers - 1:
            return self.ln_post, self.projection,
        else:
            return self.blocks[layer - 1],

    def resize_pos_embed(self, new_resolution: int | tuple[int, int]):
        """
        NOTE: call this method AFTER you load pretrained weights!
        """
        if isinstance(new_resolution, int):
            new_resolution = (new_resolution, new_resolution)
        else:
            assert len(new_resolution) == 2

        if isinstance(self._resolution, int):
            old_size = (self._resolution // self._patch_size, self._resolution // self._patch_size)
        else:
            old_size = (self._resolution[0] // self._patch_size, self._resolution[1] // self._patch_size)

        for r in new_resolution:
            assert (
                    r % self._patch_size == 0
            ), f"{new_resolution} is not divisible by {self._patch_size}"

        self._resolution = new_resolution

        with torch.no_grad():
            old_embed = self.pos_embed.data.detach()
            cls_embed, old_embed = old_embed[:1], old_embed[1:]
            new_embed = interpolate_resize_pos_embed(
                old_embed,
                old_size,
                [r // self._patch_size for r in new_resolution],
            )
            self.pos_embed = nn.Parameter(torch.cat([cls_embed, new_embed], dim=0))

    def get_embedding(self, x: torch.Tensor, full: bool = False):
        """
        Get the embedding of images (x)

        Args:
            x: image tensor
            full: whether to return the full embedding of the image patches

        Returns:
            embedding of the image
        """
        assert x.size()[-3:] == (3, *self._resolution), \
            f"input size must be (3,{self._resolution[0]},{self._resolution[1]})"

        if len(x.shape) == 5:
            bs, ts, c, h, w = x.shape
            x = x.reshape(bs * ts, c, h, w)
        elif len(x.shape) == 4:
            bs, ts = x.shape[0], None
        elif len(x.shape) == 3:
            bs, ts = None, None
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"input shape must be 3, 4 or 5 dims, got {len(x.shape)}")

        x = self.conv1(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        cls_tokens = repeat(self.cls_token, "c -> b 1 c", b=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.blocks(x)
        x = x.permute(1, 0, 2)
        if not full:
            x = x[:, 0]
        x = self.ln_post(x)

        if self.projection is not None:
            x = x @ self.projection

        if bs is None:
            x = x.squeeze(0)
        elif ts is not None:
            x = x.reshape(bs, ts, *x.shape[1:])

        return x

    def forward(self, x: torch.Tensor):
        return self.get_embedding(x, full=False)


class GPT(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            context_length: int,
            vocab_size: int,
            layers: int,
            width: int,
            heads: int,
            is_discrete_text: bool = True,
    ):
        """
        Args:
            is_discrete_text: False to use regular discrete tokens
            True for video sequence of image tokens, and `vocab_size` will be
            interpreted as the dim of each image feature.
        """
        super().__init__()
        self.context_length = context_length
        self._width = width
        self._layers = layers
        self.vocab_size = vocab_size

        self._is_discrete_text = is_discrete_text
        if is_discrete_text:
            self.token_embedding = nn.Embedding(vocab_size, width)
        else:
            self.token_embedding = nn.Linear(vocab_size, width, bias=False)
        self.pos_embed = nn.Parameter(torch.empty(self.context_length, width))
        self.blocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    width, heads, attn_mask=self.build_attention_mask()
                )
                for _ in range(layers)
            ]
        )

        self.ln_final = nn.LayerNorm(width)
        self.projection = nn.Parameter(torch.empty(width, embed_dim))

        self.initialize_parameters()
        self.layers = layers + 2

    def get_layer(self, layer: int):
        assert 0 <= layer < self.layers, f"layer {layer} is out of range"
        if layer == 0:
            return self.token_embedding, self.pos_embed
        elif layer == self.layers - 1:
            return self.ln_final, self.projection
        else:
            return self.blocks[layer - 1],

    def initialize_parameters(self):
        if self._is_discrete_text:
            nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)

        proj_std = (self._width ** -0.5) * ((2 * self._layers) ** -0.5)
        attn_std = self._width ** -0.5
        fc_std = (2 * self._width) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.projection is not None:
            nn.init.normal_(self.projection, std=self._width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def get_hidden_state(self, text: torch.Tensor, full: bool = False):
        assert text.size()[-1] == self.context_length, \
            f"input size must be (batch_size, {self.context_length})"
        assert len(text.shape) < 3, \
            f"input shape must be 1 or 2 dims, got {len(text.shape)}"

        x = self.token_embedding(text)
        if len(text.shape) == 1:
            x = x.unsqueeze(0)
        x += self.pos_embed
        x = x.permute(1, 0, 2)
        x = self.blocks(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        if not full:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        if self.projection is not None:
            x = x @ self.projection

        if len(text.shape) == 1:
            x = x.squeeze(0)

        return x

    def forward(self, text):
        return self.get_hidden_state(text, full=False)


def build_ViT(pretrained_clip=None, config_name='vit_config'):
    import yaml
    with open('config.yaml') as f:
        config = yaml.safe_load(f)[config_name]
    vit = VisionTransformer(resolution=config["pretrained_resolution"],
                            patch_size=config["patch_size"],
                            width=config["width"],
                            layers=config["layers"],
                            heads=config["heads"],
                            output_dim=config["output_dim"],
                            )
    if pretrained_clip is not None:
        image_state_dict = {}
        for k, v in pretrained_clip.named_parameters():
            if k.startswith('visual'):
                k = k.split('visual.')[1]
                if k.startswith('transformer'):
                    k = k.split('transformer.res')[1]
                elif k.startswith('pro'):
                    k = 'projection'
                elif k.startswith('pos'):
                    k = 'pos_embed'
                elif k.startswith('cla'):
                    k = 'cls_token'
                image_state_dict[k] = v.cpu()
        image_state_dict_back = vit.state_dict()
        image_state_dict_back.update(image_state_dict)
        vit.load_state_dict(image_state_dict_back)

    vit.resize_pos_embed(config["image_resolution"])
    return vit


def build_GPT(pretrained_clip=None, config_name='gpt_config'):
    import yaml
    with open('config.yaml') as f:
        config = yaml.safe_load(f)[config_name]

    gpt = GPT(embed_dim=config["embed_dim"],
              context_length=config["context_length"],
              vocab_size=config["vocab_size"],
              layers=config["layers"],
              width=config["width"],
              heads=config["heads"],
              )
    if pretrained_clip is not None:
        text_state_dict = {}
        for k, v in pretrained_clip.named_parameters():
            if k.startswith('visual') or k.startswith('log'):
                continue
            elif k.startswith('transformer'):
                k = k.split('transformer.res')[1]
            elif k.startswith('pos'):
                k = 'pos_embed'
            elif k.startswith('tex'):
                k = 'projection'
            text_state_dict[k] = v.cpu()
        text_state_dict_back = gpt.state_dict()
        text_state_dict_back.update(text_state_dict)
        gpt.load_state_dict(text_state_dict_back)
    return gpt


def build_logit_scale(pretrained_clip=None):
    if pretrained_clip is not None:
        logit_scale = nn.Parameter(pretrained_clip.logit_scale.cpu())
    else:
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    return logit_scale