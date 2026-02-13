import torch


def map_fn(preds, targets, masks=None, k=None, reduce=True):
    """
    Mean Average Precision for Information Retrieval.
    Args:
        preds:   预测评分序列，shape=[batch, seq_len]
        targets: 0-1取值的目标序列(relevance)，shape=[batch, seq_len]
        masks:   Mask，shape=[batch, seq_len]
        k:       map@k中的k，可取整数或整数列表
        reduce:  是否对batch取均值
    """
    preds = preds.clone()
    targets = targets.clone()

    seq_len = targets.shape[-1]
    preds = preds.float()
    if masks is not None:
        preds[masks] = float('-inf')

    sort_idx = preds.argsort(dim=-1, descending=True)
    rels = torch.gather(targets, dim=-1, index=sort_idx)

    if masks is not None:
        rels[masks] = 0

    idx = torch.arange(1, seq_len+1, device=targets.device).unsqueeze(0)
    p_at_k = rels.cumsum(dim=1)/idx

    if k is None:
        ks = [seq_len]
    elif isinstance(k, int):
        ks = [min(k, seq_len)]
    else:
        ks = [min(n, seq_len) for n in k]

    map_ks = []
    for n in ks:
        idx = idx[:, :n]
        maps = (p_at_k[:,:n] * rels[:,:n]).sum(dim=1)/rels[:,:n].sum(dim=1)
        maps[torch.isnan(maps)] = 0
        map_ks.append(maps.mean() if reduce else maps)

    if len(map_ks) == 1:
        map_ks = map_ks[0]
    return map_ks


if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics import average_precision_score

    row = 5
    col = 10
    p = 0
    # 如果有RuntimeWarning异常，则是因为average_precision_score不能处理标签全为0的情况
    for i in range(1):
        targets = np.random.randint(0, 2, [row, col])
        preds = np.random.random([row, col])
        m1 = average_precision_score(targets, preds, average='samples')

        padds = torch.tensor([[-1]*p] * row).float()
        preds = torch.cat([torch.tensor(preds), padds], dim=1).to(torch.float32)
        targets = torch.cat([torch.tensor(targets), padds], dim=1).to(torch.int32)

        masks = torch.tensor([[False] * col + [True] *p] * row)
        m2 = map_fn(preds, targets, masks, reduce=True)
        if (m2 - m1).abs() > 0.00001:
            print(m2 - m1)
