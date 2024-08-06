import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from functools import lru_cache


class CrossEn(nn.Module):
    def __init__(self, ):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix, gamma_vals=None):
        pt0 = sim_matrix.softmax(dim=-1) + 1e-8
        
        
        
        if gamma_vals != None:
            pt_list = []
            for i in range(len(gamma_vals)):
                #print(gamma_vals[i])
                if gamma_vals[i] != 1 and random.random()>0.6:
                    
                    pt_tmp = torch.zeros_like(pt0[i])
                    pt_tmp[1:i-1] = pt0[i,1:i-1]
                    pt_tmp[i+1:] = pt0[i,i+1:]
                    pt_tmp[0] = pt0[i,i]
                    pt_tmp[i] = pt0[i,0]
                    pt_list.append(pt_tmp)
                    #pt_list.append(pt0[i].pow(gamma_vals[i]))
                else:
                    pt_list.append(pt0[i])
                    
            pt_tmp = torch.stack(pt_list,0)
            #pt = pt_tmp.softmax(dim=-1) + 1e-8

        logpt = pt_tmp.log()
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


@lru_cache(None)
def prior_dist(x: int, device: torch.device, dtype: torch.dtype = torch.float32):
    assert x > 0, "x must be positive"
    ans = torch.ones(x, device=device, dtype=dtype)
    ed = (x + 1) // 2
    ans[:ed] = torch.as_tensor(np.linspace(0, 1, ed+1)[1:], device=device, dtype=dtype)
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

