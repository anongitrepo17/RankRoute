import torch

def mrr_fn(preds, targets, masks=None, k=None, reduce=False):
    """
    Mean Reciprocal Rank at k.
    Args:
        preds:       预测评分序列，shape=[batch, seq_len]
        targets:     one-hot目标序列(relevance)，shape=[batch, seq_len]
        masks:       Mask，shape=[batch, seq_len]
        k:           mrr@k中的k，可取整数或整数列表
        reduce:      是否对batch取均值
    """
    preds = preds.clone()
    targets = targets.clone()

    if k is None:
        atks = [targets.shape[1]]
    elif isinstance(k, int):
        atks = [k]
    else:
        atks = k

    preds = preds.float()
    if masks is not None:
        preds[masks] = float('-inf')
        targets[masks] = 0

    indices = preds.argsort(descending=True, dim=-1)
    true_sorted_by_preds = torch.gather(targets, dim=1, index=indices)

    values, indices = torch.max(true_sorted_by_preds, dim=1)
    indices = indices.type_as(values).unsqueeze(dim=0).t().expand(len(targets), len(atks))
    ats_rep = torch.tensor(data=atks, device=indices.device, dtype=torch.float32).expand(len(targets), len(atks))
    within_at_mask = (indices < ats_rep).type(torch.float32)

    result = torch.tensor(1.0) / (indices + torch.tensor(1.0))
    zero_sum_mask = torch.sum(values) == 0.0
    result[zero_sum_mask] = 0.0
    result = result * within_at_mask

    if reduce:
        return result.mean(dim=0)
    else:
        return result


if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import label_ranking_average_precision_score

    y_true=np.array([[0, 1, 0]])
    y_pred=np.array([[0.7, 0.8, 0.9]])

    m2 = label_ranking_average_precision_score(y_true, y_pred)
    print(m2)

    preds = torch.tensor(y_pred)
    targets = torch.tensor(y_true)
    m1 = mrr_fn(preds, targets)
    print(m1)
