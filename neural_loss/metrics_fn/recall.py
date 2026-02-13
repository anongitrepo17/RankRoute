import torch

def recall_fn(preds, targets, masks=None, k=None, reduce=True):
    """
    Recall.
    Args:
        preds:   预测评分序列，shape=[batch, seq_len]
        targets: 0-1取值的目标序列(relevance)，shape=[batch, seq_len]
        masks:   Mask，shape=[batch, seq_len]
        k:       map@k中的k，可取整数或整数列表
        reduce:  是否对batch取均值
    """