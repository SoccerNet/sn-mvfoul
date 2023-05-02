"""
MIT License

Copyright (c) 2022 Abdullah Hamdi

The architecture was taken from MVTorch (https://github.com/ajhamdi/mvtorch).

I changed the multi-view image architecture to a multi-view video architecture.
"""

from utils import batch_tensor, unbatch_tensor, class_freq_to_weight, torch_direction_vector, labels2freq
import torch
import numpy as np
import torchvision
from torch import dropout, nn
from mvtorch.models.voint import *

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ViewMaxAggregate(nn.Module):
    """
    A function to aggregate multi-view features using max pooling.
    """
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux


class ViewAvgAggregate(nn.Module):
    """
    A function to aggregate multi-view features using mean pooling.
    """
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.mean(aux, dim=1)
        return pooled_view.squeeze(), aux


class MVAggregate(nn.Module): 
    def __init__(self,  model, agr_type="max", feat_dim=400, lifting_net=nn.Sequential()):
        super().__init__()
        self.agr_type = agr_type

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 4)
        )


        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 8)
        )

        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
        else:
            self.aggregation_model = ViewAvgAggregate(model=model, lifting_net=lifting_net)

    def forward(self, mvimages):

        pooled_view, attention = self.aggregation_model(mvimages)

        inter1 = self.inter(pooled_view)
        pred_action = self.fc_action(inter1)
        pred_offence_severity = self.fc_offence(inter1)

        return pred_offence_severity, pred_action, attention
