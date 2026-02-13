from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


# class _RankDataset(Dataset):
#     def __init__(self, model_scores, target_ranks, ranking_values, padd_idx=None):
#         super().__init__()
#         assert len(model_scores) == len(target_ranks) == len(ranking_values)

#         self.model_scores = model_scores
#         self.target_ranks = target_ranks
#         self.ranking_values = ranking_values
#         if padd_idx is None:
#             self.padd_idx = max(max(r) for r in self.target_ranks) + 1
#         else:
#             self.padd_idx = padd_idx

#     def __len__(self):
#         return len(self.model_scores)

#     def __getitem__(self, index):
#         return self.model_scores[index], self.target_ranks[index], self.ranking_values[index]


# class _RankDataLoader(DataLoader):
#     def __init__(self, dataset, batch_size, shuffle):
#         super().__init__(dataset, batch_size, shuffle, collate_fn=self._collate)
#         self.padd_idx = dataset.padd_idx

#     def _collate(self, batch):
#         model_scores, target_ranks, ranking_values, masks = self.padd(batch)
#         model_scores = torch.tensor(model_scores, dtype=torch.float32)
#         target_ranks = torch.tensor(target_ranks, dtype=torch.long)
#         masks = torch.tensor(masks)
#         ranking_values = torch.tensor(ranking_values, dtype=torch.float32)
#         return model_scores.unsqueeze(-1), target_ranks, masks, ranking_values

#     def padd(self, batch):
#         batch_len = max(len(data[0]) for data in batch)
#         model_scores = []
#         target_ranks = []
#         ranking_values = []
#         masks = []
#         for s, r, v in batch:
#             data_len = len(s)
#             model_scores.append(np.pad(s, (0, batch_len-data_len), "constant", constant_values=-1))
#             target_ranks.append(np.pad(r, (0, batch_len-data_len), "constant", constant_values=self.padd_idx))
#             ranking_values.append(v)
#             masks.append([False] * data_len + [True]*(batch_len-data_len))
#         return np.array(model_scores), np.array(target_ranks), np.array(ranking_values), np.array(masks)


class RankDataset(Dataset):
    def __init__(self, data_list, padd_idx=None):
        super().__init__()
        self.data_list = data_list
        if padd_idx is None:
            self.padd_idx = max(max(r[1])for rs in self.data_list for r in rs) + 1
        else:
            self.padd_idx = padd_idx

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


class RankDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, collate_fn=None, **kwargs):
        super().__init__(dataset, batch_size, shuffle, collate_fn=self._collate, **kwargs)
        self.padd_idx = dataset.padd_idx

    def _collate(self, batch):
        model_scores, target_ranks, ranking_values, masks = self.padd(batch)
        model_scores = torch.tensor(model_scores, dtype=torch.float32)
        target_ranks = torch.tensor(target_ranks, dtype=torch.int32)
        masks = torch.tensor(masks)
        ranking_values = torch.tensor(ranking_values, dtype=torch.float32)
        return model_scores.unsqueeze(-1), target_ranks, masks, ranking_values

    def padd(self, batch):
        batch_len = max(len(d[0]) for datas in batch for d in datas)
        model_scores = []
        target_ranks = []
        ranking_values = []
        masks = []
        for datas in batch:
            for s, r, v in datas:
                data_len = len(s)
                model_scores.append(np.pad(s, (0, batch_len-data_len), "constant", constant_values=-1))
                target_ranks.append(np.pad(r, (0, batch_len-data_len), "constant", constant_values=self.padd_idx))
                ranking_values.append(v)
                masks.append([False] * data_len + [True]*(batch_len-data_len))
        return np.array(model_scores), np.array(target_ranks), np.array(ranking_values), np.array(masks)


if __name__ == "__main__":
    from neural_loss.data_prepare import load_dataset
    min_seq_len = 30
    max_seq_len = 200
    num_seqs = 10000
    repeat = 10
    datas = load_dataset('dcg', num_seqs, repeat, min_seq_len, max_seq_len, False, 5, 'train')  # spearman kendall dcg
    # ds = RankDataset(datas)
    # dl = RankDataLoader(ds, 64, False)
    # for scores, ranks, masks, values in dl:
    #     print(scores, ranks, masks, values)
