import torch


class NDCG:
    def __init__(self, k=None, exponential=False, reduce=True):
        """
        Args:
            k:           ndcg@k中的k，可取整数或整数列表
            exponential: 是否对目标序列取2^{relevance}-1
            reduce:      是否对batch取均值
        """
        super().__init__()
        self.k = k
        self.exponential = exponential
        self.reduce = reduce
        self.name = 'ndcg'

    def __call__(self, preds, targets, masks=None):
        """
        Args:
            preds:   预测评分序列，shape=[batch, seq_len]
            targets: 目标序列(relevance)，shape=[batch, seq_len]
            masks:   Mask，shape=[batch, seq_len]
        """
        ndcg = ndcg_fn(preds, targets, masks, self.k, self.exponential, reduce=self.reduce)
        if not isinstance(ndcg, list):
            return ndcg
        else:
            return {f'@{k}': m for k, m in zip(self.k, ndcg)}


def ndcg_fn(preds, targets, masks=None, k=None, exponential=False, reduce=True):
    """
    Normalized Discounted Cumulative Gain.
    Args:
        preds:       预测评分序列，shape=[batch, seq_len]
        targets:     目标序列(relevance)，shape=[batch, seq_len]
        masks:       Mask，shape=[batch, seq_len]
        k:           ndcg@k中的k，可取整数或整数列表
        exponential: 是否对目标序列取2^{relevance}-1
        reduce:      是否对batch取均值
    """
    preds = preds.clone()
    targets = targets.clone()

    dcgs = dcg_fn(preds, targets, masks, k, exponential)
    idcgs = dcg_fn(targets, targets, masks, k, exponential)
    ndcgs = [dcg/idcg for dcg, idcg in zip(dcgs, idcgs)]

    # idcg可能等于0
    for ndcg in ndcgs:
        ndcg[torch.isnan(ndcg)] = 0

    if reduce:
        ndcgs = [ndcg.mean() for ndcg in ndcgs]
    if len(ndcgs) == 1:
        ndcgs = ndcgs[0]
    return ndcgs


def dcg_fn(preds, targets, masks=None, k=None, exponential=False):
    seq_len = targets.shape[-1]
    preds = preds.float()
    if masks is not None:
        preds[masks] = float('-inf')

    sort_idx = preds.argsort(dim=-1, descending=True)
    rels = torch.gather(targets, dim=-1, index=sort_idx)
    if exponential:
        rels = torch.exp2(rels) -1
    if masks is not None:
        rels[masks] = 0

    idx = torch.arange(1, seq_len+1, device=targets.device)
    discount = 1/(idx+1).log2()

    dcg = rels * discount.unsqueeze(0)

    if k is None:
        ks = [seq_len]
    elif isinstance(k, int):
        ks = [min(k, seq_len)]
    else:
        ks = [min(n, seq_len) for n in k]

    dcg_atn = []
    for n in ks:
        _dcg = dcg[:, :n]
        dcg_atn.append(_dcg.sum(-1))
    return dcg_atn


if __name__ == "__main__":
    from torchmetrics import RetrievalNormalizedDCG
    import numpy as np

    row = 40
    col = 50
    k=5
    p = 5
    for i in range(1000):
        preds = torch.tensor(np.random.random(size=[row, col]))
        targets = torch.tensor(np.random.randint(low=0, high=5, size=[row, col]))
        targets[0] = 0
        masks = torch.tensor([[False]*col + [True]*p] * row)
        padds = torch.tensor([[-1]*p] * row)
        m1 = ndcg_fn(torch.cat([preds, padds], dim=1), torch.cat([targets, padds], dim=1), masks, reduce=True, k=k)

        preds = preds.view(-1)
        targets = targets.view(-1)
        indxes = torch.LongTensor(np.arange(row).repeat(col))
        m2=RetrievalNormalizedDCG(k=k)(preds, targets, indexes=indxes)

        if (m1 - m2).abs() > 0.00001:
            print(m1, m2, m1-m2)
