import numpy as np
from random import randint
import random as rand
from scipy.stats import spearmanr, truncnorm, kendalltau
from os import path as osp
from functools import partial
import pickle
import multiprocessing as mp


# @njit
def one_score_sequence(seq_len, lower=0, upper=1):
    """
    Randomly generate a sequence of length seq_len from four distributions
    """
    type_rand = randint(0, 8)
    norm = truncnorm(lower, upper, loc=0, scale=1)

    if type_rand == 0:
        rand_seq = np.random.rand(seq_len) * (upper-lower) + lower
    elif type_rand == 1:
        rand_seq = np.random.uniform(lower, upper, seq_len)
    elif type_rand == 2:
        rand_seq = norm.rvs(seq_len)
    elif type_rand == 3:
        a = np.random.rand() * (upper-lower) + lower
        b = np.random.rand() * (upper-lower) + lower
        rand_seq = np.arange(a, b, (b - a) / seq_len)[:seq_len]
    elif type_rand == 4:
        split = randint(1, seq_len)
        rand_seq = np.zeros(seq_len)
        rand_seq[:split] = np.random.rand(split) * (upper-lower) + lower
        rand_seq[split:] = norm.rvs(seq_len - split)
    elif type_rand == 5:
        split = randint(1, seq_len)
        rand_seq = np.zeros(seq_len)
        rand_seq[:split] = np.random.uniform(lower, upper, split)
        rand_seq[split:] = norm.rvs(seq_len - split)
    elif type_rand == 6:
        split = randint(1, seq_len)
        rand_seq = np.zeros(seq_len)
        rand_seq[:split] = np.random.rand(split) * (upper-lower) + lower
        rand_seq[split:] = np.random.uniform(lower, upper, seq_len - split)
    elif type_rand == 7:
        split = randint(1, seq_len)
        a = np.random.rand() * (upper-lower) + lower
        b = np.random.rand() * (upper-lower) + lower
        half_rand_seq = np.arange(a, b, (b - a) / split)
        np.random.shuffle(half_rand_seq)
        rand_seq = np.zeros(seq_len)
        rand_seq[:split] = half_rand_seq[:split]
        rand_seq[split:] = np.random.rand(seq_len - split) * (upper-lower) + lower
    elif type_rand == 8:
        rand_seq = np.arange(lower, upper, (upper - lower) / seq_len)[:seq_len]
    return rand_seq.astype(np.float32)


# @njit
def shuffle(seq):
    """Return a new, shuffled seq"""
    new_seq = seq.copy()
    np.random.shuffle(new_seq)
    return new_seq


def get_random_lens(size, min_len=30, max_len=200):
    lens = np.random.lognormal(mean=4.78, sigma=0.46, size=int(size)).astype(np.int32)
    lens[lens<min_len] = min_len
    lens[lens>max_len] = max_len
    return lens.tolist()


def get_metric(m_name):
    if m_name == 'spearman':
        return spearmanr_fn
    if m_name == 'kendall':
        return kendall_fn
    if m_name == 'dcg':
        return dcg_fn
    if m_name == 'avg_recall':
        return average_recall_at_k
    raise ValueError('metric name error!')


def average_recall_at_k(predicted_scores, true_ranking, k):
    """
    Example:
        predicted_scores = np.array([0.1, 0.5, 0.4, 0.3, 0.2])
        true_ranking = np.array([0, 1, 1, 0, 0])

        ar2 = average_recall_at_k(predicted_scores, true_ranking, 2)
        ar3 = average_recall_at_k(predicted_scores, true_ranking, 3)

        ar2 is 1.0
        ar3 is 0.8889
    """
    # Sort according to the score list and obtain the indices of the top max_k documents
    top_k_idx = np.argsort(predicted_scores)[-k:]

    # Calculate the total number of relevant documents G
    G = sum(true_ranking > 0)
    if G == 0:
        return 0

    # Calculate the sum of recall@k
    recall_sum = 0
    for rank in range(1, k + 1):
        recall_sum += sum(true_ranking[top_k_idx[-rank:]] > 0)/min(rank, G)

    return recall_sum/k


def spearmanr_fn(s, r):
    return spearmanr(s, r)[0]

def kendall_fn(s, r):
    return kendalltau(s, r)[0]

def dcg_fn(preds, targets, k=None, exponential=False):
    """
    Discounted Cumulative Gain (DCG)
    """
    seq_len = len(targets)
    sort_idx = (-preds).argsort(axis=-1)
    rels = targets[sort_idx]
    if exponential:
        rels = np.exp2(rels) -1

    idx = np.arange(1, seq_len+1)
    discount = 1/np.log2(idx+1)

    dcg = rels * discount
    if k is not None:
        return dcg[:k].sum()
    else:
        return dcg.sum()


def generate_dataset(metric, num_seqs, repeat, min_len, max_len, random_len, target_type, at_k):
    """
    Generate predicted sequences and target sequences.
    """
    assert metric in ['spearman', 'kendall', 'dcg', 'avg_recall']
    metric = get_metric(metric)

    if random_len:
        lens = get_random_lens(num_seqs * (max_len - min_len + 1), min_len, max_len)
    else:
        lens =list(range(min_len, max_len + 1)) * int(num_seqs)
        rand.shuffle(lens)

    task = partial(seq_task, metric=metric, repeat=repeat, target_type=target_type, at_k=at_k)
    if len(lens) > 10000:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            datalst = pool.map(task, lens, 500)
    else:
        datalst = list(map(task, lens))

    # model_scores, target_ranks, ranking_values = list(zip(*list(chain(*datalst))))
    # return model_scores, target_ranks, ranking_values

    return datalst


def seq_task(seq_len, metric, repeat, target_type, at_k):
    # score sequence
    scores = one_score_sequence(seq_len)
    # ranking sequence
    if target_type == 'full':
        ranks = np.arange(1, seq_len + 1)
    elif target_type == 2:
        ranks = np.random.choice([0, 1], size=seq_len)
    elif target_type == 3:
        p = [0.75, 0.2, 0.05]
        ranks = np.random.choice([0, 1, 2], size=seq_len, p=p)
    elif target_type == 5:
        p = [0.515, 0.325, 0.134, 0.018, 0.008]
        ranks = np.random.choice([0, 1, 2, 3, 4], size=seq_len, p=p)

    repeat_scores = [shuffle(scores) for _ in range(repeat)]
    repeat_ranks = [shuffle(ranks) for _ in range(repeat)]
    if target_type != 'full':
        sf_ranks = [shuffle(ranks) for _ in range(repeat)]
        repeat_scores.extend([ranks/target_type for ranks in sf_ranks])
        repeat_ranks.extend(sf_ranks)

    if at_k is None:
        return [(s.astype(np.float32), r.astype(np.int16), metric(s, r)) for s, r in zip(repeat_scores, repeat_ranks)]
    else:
        return [(s.astype(np.float32), r.astype(np.int16), metric(s, r, at_k)) for s, r in zip(repeat_scores, repeat_ranks)]


def load_dataset(metric, num_seqs, repeat, min_len, max_len, random_len, target_type, stage, path='./neural_loss/datasets', save=True, at_k=None, logger=None):
    """
    Load dataset and return (model_scores, target_ranks, ranking_values).
    If the data already exists, return it directly; otherwise, generate and save the data before returning.
    """

    # "full" indicates that the rank is the entire sequence, "2" indicates the values are 0, 1, "3" indicates the values are 0, 1, 2, and "5" indicates the values are 0, 1, 2, 3, 4
    assert target_type in ['full', 2, 3, 5]
    path = osp.join(path, f'ds-{metric}-{min_len}_{max_len}_{num_seqs}_{repeat}_{"randT" if random_len else "randF"}_{target_type}-{stage}.pkl')
    if osp.exists(path):
        if logger:
            logger.info(f'loading {stage} dataset for loss model ...')
        else:
            print(f'loading {stage} dataset for loss model ...')
        with open(path, "rb") as f:
            # model_scores, target_ranks, ranking_values = pickle.load(f)
            ds = pickle.load(f)
    else:
        # model_scores, target_ranks, ranking_values = generate_dataset(metric, num_seqs, repeat, min_len, max_len, random_len, target_type)
        if logger:
            logger.info(f'generating {stage} dataset for loss model ...')
        else:
            print(f'generating {stage} dataset for loss model ...')
        ds = generate_dataset(metric, num_seqs, repeat, min_len, max_len, random_len, target_type, at_k)
        if save:
            with open(path, "wb") as f:
                pickle.dump(ds, f)
    return ds


if __name__ == "__main__":
    min_seq_len = 30
    max_seq_len = 35
    num_seqs = 100
    repeat = 10
    print(f'Total data volume: {(max_seq_len-min_seq_len+1)*num_seqs*repeat}')
    import time
    start = time.time()
    ds = load_dataset('dcg', num_seqs, repeat, min_seq_len, max_seq_len, False, 3, 'train')  # spearman kendall dcg
    print('time:', time.time()-start)
