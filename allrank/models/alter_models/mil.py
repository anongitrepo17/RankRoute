'''
Attention-based Deep Multiple Instance Learning
'''

import torch
from torch import nn
from torch.nn import init


class MIL(nn.Module):
    def __init__(self, feat_dim, max_set_size, gate):
        super().__init__()
        if gate:
            self.att = MILGateAtt(max_set_size, feat_dim)
        else:
            self.att = MILAtt(max_set_size, feat_dim)

    def forward(self, inputs, masks, _):
        masks = torch.logical_not(masks)

        att_weight = self.att(inputs, masks)
        return inputs * att_weight


class MILAtt(nn.Module):
    """
    原文式（8）的实现
    """
    def __init__(self, max_len, data_dim):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(max_len, 1))
        self.V = nn.Parameter(torch.Tensor(max_len, data_dim))
        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.V)

    def forward(self, H, mask):
        scores = self.w.transpose(0, 1) @ torch.tanh(self.V @ H.transpose(2, 1))
        scores = scores.squeeze(1)
        scores = scores.masked_fill(mask == False, -1e9)
        return scores.softmax(1).unsqueeze(-1)


class MILGateAtt(nn.Module):
    """
    原文式（9）的实现
    """
    def __init__(self, max_len, data_dim):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(max_len, 1))
        self.V = nn.Parameter(torch.Tensor(max_len, data_dim))
        self.U = nn.Parameter(torch.Tensor(max_len, data_dim))
        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.V)
        init.xavier_uniform_(self.U)

    def forward(self, H, mask):
        scores = self.w.transpose(0, 1) @ (torch.tanh(self.V @ H.transpose(2, 1)) * torch.sigmoid(self.U @ H.transpose(2, 1)))
        scores = scores.squeeze(1)
        scores = scores.masked_fill(mask == False, -1e9)
        return scores.softmax(1).unsqueeze(-1)
