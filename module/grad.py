import numpy as np
import torch
import torch.nn as nn

from functools import lru_cache


class CrossEn(nn.Module):
    def __init__(self, ):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        pt = sim_matrix.softmax(dim=-1) + 1e-8  # [B,B]
        logpt = pt.log()
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


@lru_cache(None)
def prior_dist(x: int, device: torch.device, dtype: torch.dtype = torch.float32):
    assert x > 0, "x must be positive"
    ans = torch.ones(x, device=device, dtype=dtype)
    ed = (x + 1) * 3 // 4
    ans[:ed] = torch.as_tensor(np.linspace(0.4, 1, ed+1)[1:], device=device, dtype=dtype)
    return ans.log()


class MultiCrossEn(nn.Module):
    def __init__(self, ):
        super(MultiCrossEn, self).__init__()

    def forward(self, sim_matrix):
        _, N, _ = sim_matrix.shape
        pt = sim_matrix.softmax(dim=-1) + 1e-8  # [B,N,B]
        logpt = pt.log()
        logpt = torch.diagonal(logpt, dim1=0, dim2=2).t()  # [B,N]

        pd_logpt = prior_dist(N, device=sim_matrix.device, dtype=sim_matrix.dtype)  # [N]
        nce_loss = ((pd_logpt - logpt).abs())  # [B,N]
        sim_loss = nce_loss.mean()
        return sim_loss


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        world_size = torch.distributed.get_world_size()
        local_rank = torch.distributed.get_rank()

        output = [torch.empty_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(output, tensor)

        ctx.rank = local_rank
        ctx.batch_size = tensor.shape[0]

        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
        )
