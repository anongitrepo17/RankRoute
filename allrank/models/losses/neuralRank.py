import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    """
    输入分数和排名张量，输出相关性系数，来近似斯皮尔曼相关系数的计算
    Args:
        hidden_dim：隐藏层维度大小。
        num_heads：注意力头的数量。
        num_layers：编码器的层数。
    """

    def __init__(self, model_dim, num_heads, num_layers, embedding_num, padd_idx, out_active):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        if out_active == 'sigmoid':
            self.out_act = F.sigmoid
        elif out_active == 'tanh':
            self.out_act = F.tanh
        elif out_active == 'relu':
            self.out_act = F.relu
        else:
            raise ValueError("out_active must be 'sigmoid', 'tanh' or 'relu'")

        self.scores_trans = nn.Linear(1, model_dim)
        self.ranks_embes = nn.Embedding(embedding_num, model_dim, padding_idx=padd_idx)

        self.transormer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim, batch_first=False, dropout=0),
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        self.out = nn.Linear(model_dim, 1)

    def forward(self, scores, ranks, masks):
        outputs = self.scores_trans(scores) + self.ranks_embes(ranks)
        outputs = self.transormer(outputs.transpose(0, 1), src_key_padding_mask=masks)
        outputs = self.out(outputs.transpose(0, 1))

        outputs = outputs.squeeze(-1) * (1-masks.float())
        return self.out_act(outputs.sum(dim=1)/(1-masks.float()).sum(dim=1))


class NeuralRankNDCG:
    def __init__(self, model_path, model_dim, num_heads, num_layers, embedding_num, padd_idx, out_active):
        super().__init__()
        self.padd_idx = padd_idx
        self.dcg = Model(model_dim, num_heads, num_layers, embedding_num, padd_idx, out_active)
        self.dcg.load_state_dict(torch.load(model_path, weights_only=True))
        for param in self.dcg.parameters():
            param.requires_grad = False

    def __call__(self, preds, targets):
        self.dcg.to(preds.device)
        preds = preds.unsqueeze(-1)
        masks = targets == -1
        targets[masks] = self.padd_idx
        dcg = self.dcg(preds, targets.long(), masks)
        idcg = self.dcg((targets/self.padd_idx).unsqueeze(-1), targets.long(), masks)
        ndcg = (dcg/idcg).nan_to_num(0)  # idcg可能为0

        return -ndcg.mean()


class NeuralRankRecall:
    def __init__(self, model_path, model_dim, num_heads, num_layers, embedding_num, padd_idx, out_active):
        super().__init__()
        self.padd_idx = padd_idx
        self.recall = Model(model_dim, num_heads, num_layers, embedding_num, padd_idx, out_active)
        self.recall.load_state_dict(torch.load(model_path, weights_only=True))
        for param in self.recall.parameters():
            param.requires_grad = False

    def __call__(self, preds, targets):
        self.recall.to(preds.device)
        preds = preds.unsqueeze(-1)
        masks = targets == -1
        targets[masks] = self.padd_idx
        recall = self.recall(preds, targets.long(), masks)

        return -recall.mean()


neural_rank_ndcg_MQ = NeuralRankNDCG('loss_model_MQ.pth',
                                     model_dim = 64,          # 模型维度
                                     num_heads = 4,           # 头数
                                     num_layers = 2,          # 层数
                                     embedding_num = 4,       # embedding数量，MQ为 4（[0,1,2,3]），WEB3K中为6（[0,1,2,3,4,5]）
                                     padd_idx = 3,            # embedding的最后一个数字，MQ为3，WEB3K中为5
                                     out_active = 'relu'     # 激活函数，dcg为relu，spearman为tanh
                                    )


# neural_rank_ndcg_WEB = NeuralRankNDCG('loss_model_WEB.pth',
#                                      model_dim = 64,          # 模型维度
#                                      num_heads = 4,           # 头数
#                                      num_layers = 2,          # 层数
#                                      embedding_num = 6,       # embedding数量，MQ为 4（[0,1,2,3]），WEB3K中为6（[0,1,2,3,4,5]）
#                                      padd_idx = 5,            # embedding的最后一个数字，MQ为3，WEB3K中为5
#                                      out_active = 'relu'     # 激活函数，dcg为relu，spearman为tanh
#                                     )


neural_rank_recall_MQ = NeuralRankRecall('loss_model_51-51-5000_@10.pth',
                                     model_dim = 64,          # 模型维度
                                     num_heads = 8,           # 头数
                                     num_layers = 2,          # 层数
                                     embedding_num = 3,       # embedding数量，MSCOCO为3，MQ为 4（[0,1,2,3]），WEB3K中为6（[0,1,2,3,4,5]）
                                     padd_idx = 2,            # embedding的最后一个数字，MSCOCO为2，MQ为3，WEB3K中为5
                                     out_active = 'sigmoid'     # 激活函数，dcg为relu，spearman为tanh
                                    )
