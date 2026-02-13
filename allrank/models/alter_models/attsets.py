'''
Robust Attentional Aggregation of Deep Feature Sets for Multi-view 3D Reconstruction
'''

import torch
from torch import nn
from torch.nn import init


class AttSets(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.att = Attention(feat_dim)

    def forward(self,  inputs, masks, _):
        masks = torch.logical_not(masks)

        att_weight = self.att(inputs, masks)
        outputs = inputs * att_weight
        return outputs


class Attention(nn.Module):
    """
    原文式（2）-（3）的实现
    """
    def __init__(self, data_dim):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(data_dim, data_dim))
        init.xavier_uniform_(self.W)

    def forward(self, H, mask):
        scores = H @ self.W
        scores = scores.masked_fill(mask.unsqueeze(-1) == False, -1e9)
        return scores.softmax(1)
