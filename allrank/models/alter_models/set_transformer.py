import torch
from torch import nn
import torch.nn.functional as F
import math


class SetTransformer(nn.Module):
    def __init__(self, model_dim, head_nums, layer_nums):
        super().__init__()
        self.encs = nn.ModuleList([SAB(model_dim= model_dim, num_heads=head_nums) for _ in range(layer_nums)])

    def forward(self, inputs, masks, _):
        masks = torch.logical_not(masks).unsqueeze(1).unsqueeze(1)
        outputs = self.encs[0](inputs, masks)
        for layer in self.encs[1:]:
            outputs = layer(outputs, masks)
        return outputs


class SAB(nn.Module):
    def __init__(self, model_dim, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(model_dim, num_heads, layer_norm=ln)

    def forward(self, X, mask):
        return self.mab(X, X, mask)


class MAB(nn.Module):
    def __init__(self, model_dim, num_heads, layer_norm=False, dropout=.0):
        super().__init__()
        self.attention = Attention(model_dim, model_dim, num_heads, dropout)
        self.skip_att = SkipConnection(model_dim if layer_norm else None, dropout)
        self.ff = FeedForward(model_dim, dropout)
        self.skip_ff = SkipConnection(model_dim if layer_norm else None, dropout)

    def forward(self, Q, KV, mask=None):
        Z = self.attention(Q, KV, KV, mask)
        Z = self.skip_att(Q, Z)
        Z_ = self.ff(Z)
        Z = self.skip_ff(Z, Z_)
        return Z


class PMA(nn.Module):
    def __init__(self, model_dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, model_dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(model_dim, num_heads, layer_norm=ln)

    def forward(self, X, mask):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, mask)


class Attention(nn.Module):
    def __init__(self, dim_QK, dim_V, num_heads, dropout=.0):
        super().__init__()
        assert dim_QK % num_heads == 0, "Mismatching Dimensions!"
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.head_dim_qk = dim_QK // num_heads
        self.head_dim_v = dim_V // num_heads

        self.fc_q = nn.Linear(dim_QK, dim_V)
        self.fc_k = nn.Linear(dim_QK, dim_V)
        self.fc_v = nn.Linear(dim_QK, dim_V)
        self.fc_out = nn.Linear(dim_V, dim_V)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        batch_size = Q.shape[0]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim_qk).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim_qk).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim_v).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_V)
        if mask is not None:
            scores = scores.masked_fill(mask == False, -1e9)
        atten = scores.softmax(dim=-1)

        atten = self.dropout(atten)

        Z = torch.matmul(atten, V)
        Z = Z.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_V)
        del Q
        del K
        del V
        return self.fc_out(Z)


class SkipConnection(nn.Module):
    def __init__(self, norm_dim=None, dropout=.0):
        super().__init__()
        self.layer_norm = None if norm_dim is None else nn.LayerNorm(norm_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, new_X):
        Z = self.dropout(new_X)
        Z = X + Z
        Z = Z if self.layer_norm is None else self.layer_norm(Z)
        return Z


class FeedForward(nn.Module):
    def __init__(self, model_dim, dropout=.0):
        super().__init__()
        self.ff_1 = nn.Linear(model_dim, 4*model_dim)
        self.ff_2 = nn.Linear(4*model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        Z = F.relu(self.ff_1(X))
        Z = self.dropout(Z)
        Z = F.relu(self.ff_2(Z))
        return Z

