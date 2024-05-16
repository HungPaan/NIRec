import random
import torch
import numpy as np
from torch_geometric.utils import (
    scatter,
    segment,
)
from typing import Optional
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def softmax_1(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
) -> Tensor:

    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        count = ptr[1:] - ptr[:-1]
        ptr = ptr.view(size)
        src_max = segment(src.detach(), ptr, reduce='max')
        src_max = src_max.repeat_interleave(count, dim=dim)
        out = (src - src_max).exp()
        out_sum = segment(out, ptr, reduce='sum') + 1e-16
        out_sum = out_sum.repeat_interleave(count, dim=dim)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
        src_max_per_edge = torch.max(src_max.index_select(dim, index), torch.zeros_like(src))
        out = src - src_max_per_edge

        zeros_tensor = torch.zeros_like(src) - src_max_per_edge
        out = out.exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-16
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError
    #print('out', out, out.shape, out_sum, out_sum.shape)
    return out / (out_sum + zeros_tensor.exp())


def check_and_create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"create success")
    else:
        print(f"existed")