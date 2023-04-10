import torch.nn as nn
from x_transformers.x_transformers import Encoder, ContinuousTransformerWrapper


class SequenceTransformer(nn.Module):
    def __init__(
            self,
            *,
            input_dim: int,
            embed_dim: int = None,
            depth: int,
            num_heads: int,
            max_seq_len: int,
            # ----- extra tricks, see x_transformers repo ----
            ff_glu=True,
            ff_swish=True,
            attn_one_kv_head=False,
            rel_pos_bias=False,
    ):
        """
        Reference arch:
            bert_base:
                embed_dim = 768
                depth = 12
                num_heads = 12
            bert_large:
                embed_dim = 1024
                depth = 24
                num_heads = 16
        Args:
            input_dim: continuous input feature dimension
            max_seq_len: max sequence length
            embed_dim: embedding dimension, if None, then it is the same as input_dim
                BUT will not add a projection layer from input -> first embedding
                if embed_dim is specified, a projection layer will be added even if
                input_dim == embed_dim
        """
        super().__init__()
        assert isinstance(max_seq_len, int)
        assert isinstance(input_dim, int)
        assert isinstance(depth, int)
        assert isinstance(num_heads, int)

        self.model = ContinuousTransformerWrapper(
            max_seq_len=max_seq_len,
            attn_layers=Encoder(
                dim=input_dim if embed_dim is None else embed_dim,
                depth=depth,
                heads=num_heads,
                ff_glu=ff_glu,
                ff_swish=ff_swish,
                attn_one_kv_head=attn_one_kv_head,
                rel_pos_bias=rel_pos_bias,
            ),
            # if embed_dim is None, do NOT add an input feature projection layer
            dim_in=None if embed_dim is None else input_dim,
            dim_out=None,
        )
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.num_heads = num_heads
        self.layers = 1

    def get_layer(self, layer: int):
        assert layer == 0
        return self.model,

    @property
    def output_dim(self):
        return self.input_dim if self.embed_dim is None else self.embed_dim

    def get_temporal_embedding(self, x, mask=None, full=False):
        B, L, F = x.size()
        x = self.model(x, mask=mask)
        if full:
            assert x.shape == (B, L, self.output_dim)
        else:
            x = x.mean(dim=1)
            assert x.shape == (B, self.output_dim)
        return x

    def forward(self, x, mask=None):
        return self.get_temporal_embedding(x, mask=mask, full=False)


def build_sequence_encoder(config_name):
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)[config_name]
    tt = SequenceTransformer(input_dim=config['input_dim'],
                             embed_dim=config['embed_dim'],
                             depth=config['depth'],
                             num_heads=config['num_heads'],
                             max_seq_len=config['max_seq_len'],
                             ff_glu=config['ff_glu'],
                             ff_swish=config['ff_swish'],
                             attn_one_kv_head=config['attn_one_kv_head'],
                             rel_pos_bias=config['rel_pos_bias'])
    return tt