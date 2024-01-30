import torch
import matplotlib.pyplot as plt
import imageio
import numpy as np
import pickle
import os

"""
MIT License

Copyright (c) 2022 Abdullah Hamdi

The architecture was taken from MVTorch (https://github.com/ajhamdi/mvtorch).

I changed the multi-view image architecture to a multi-view video architecture.
"""

def batch_tensor(tensor, dim=1, squeeze=False):
    """
    A function to reshape PyTorch tensor `tensor` along some dimension `dim` to the batch dimension 0 such that the tensor can be processed in parallel. 
    If `sqeeze`=True, the dimension `dim` will be removed completely, otherwise it will be of size=1. Check `unbatch_tensor()` for the reverese function.
    """
    batch_size, dim_size = tensor.shape[0], tensor.shape[dim]
    returned_size = list(tensor.shape)
    returned_size[0] = batch_size * dim_size
    returned_size[dim] = 1
    if squeeze:
        return tensor.transpose(0, dim).reshape(returned_size).squeeze_(dim)
    else:
        return tensor.transpose(0, dim).reshape(returned_size)


def unbatch_tensor(tensor, batch_size, dim=1, unsqueeze=False):
    """
    A function to chunk pytorch tensor `tensor` along the batch dimension 0 and concatenate the chuncks on dimension `dim` to recover from `batch_tensor()` function.
    If `unsqueee`=True, it will add a dimension `dim` before the unbatching. 
    """
    fake_batch_size = tensor.shape[0]
    nb_chunks = int(fake_batch_size / batch_size)
    if unsqueeze:
        return torch.cat(torch.chunk(tensor.unsqueeze_(dim), nb_chunks, dim=0), dim=dim).contiguous()
    else:
        return torch.cat(torch.chunk(tensor, nb_chunks, dim=0), dim=dim).contiguous()

