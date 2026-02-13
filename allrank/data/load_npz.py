from .dataset_loading import LibSVMDataset, fix_length_to_longest_slate
import numpy as np
from os import path as osp


def load_npz_dataset(input_path, validation_ds_role):
    train_ds = load_ds('train', input_path)
    val_ds = load_ds(validation_ds_role, input_path)
    return train_ds, val_ds


def load_ds(role, input_path):
    data = np.load(osp.join(input_path, f'{role}.npz'))
    query_embedds=data['query_embedds']
    labels=data['labels']
    object_embedds=data['object_embedds']

    query_ids = np.arange(labels.shape[0]).repeat(labels.shape[1])

    query_embedds = query_embedds.repeat(labels.shape[1], axis=0)
    labels = labels.reshape(-1)
    object_embedds = object_embedds.reshape(-1, object_embedds.shape[-1])

    ds = LibSVMDataset(np.concatenate([query_embedds, object_embedds], axis=1), labels, query_ids)
    ds.transform = fix_length_to_longest_slate(ds)
    return ds
