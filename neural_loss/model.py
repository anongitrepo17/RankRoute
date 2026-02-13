import torch.nn as nn
from torch.nn import functional as F
from functools import partial

class Model(nn.Module):
    def __init__(self, model_dim, num_heads, num_layers, embedding_num, padd_idx, out_active):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.padd_idx = padd_idx

        if out_active == 'sigmoid':
            self.out_act = F.sigmoid
        elif out_active == 'tanh':
            self.out_act = F.tanh
        elif out_active == 'relu':
            self.out_act =  partial(F.leaky_relu, negative_slope=0.00001)
        else:
            raise ValueError("out_active must be 'sigmoid', 'tanh' or 'relu'")

        self.scores_trans = nn.Linear(1, model_dim)
        self.ranks_embes = nn.Embedding(embedding_num, model_dim, padding_idx=padd_idx)

        self.transormer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim, batch_first=True, dropout=0),
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        self.out = nn.Linear(model_dim, 1)

    def forward(self, scores, ranks, masks):
        outputs = self.scores_trans(scores) + self.ranks_embes(ranks)
        outputs = self.transormer(outputs, src_key_padding_mask=masks)
        outputs = self.out(outputs)
        outputs = outputs.squeeze(-1) * (1-masks.float())
        return self.out_act(outputs.sum(dim=1)/(1-masks.float()).sum(dim=1))
