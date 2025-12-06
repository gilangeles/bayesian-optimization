import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split


def get_splits(len_dataset: int, val_split: int | float):
    """Computes split lengths for train and validation set."""
    if isinstance(val_split, int):
        train_len = len_dataset - val_split
        splits = [train_len, val_split]
    elif isinstance(val_split, float):
        val_len = int(val_split * len_dataset)
        train_len = len_dataset - val_len
        splits = [train_len, val_len]
    else:
        raise ValueError(f"Unsupported type {type(val_split)}")

    return splits


def split_dataset(
    dataset: Dataset,
    val_split: float | int = 0.2,
    train: bool = True,
    generator: torch.Generator | None = None,
):
    """Splits the dataset into train and validation set."""
    len_dataset = len(dataset)
    splits = get_splits(len_dataset, val_split)
    dataset_train, dataset_val = random_split(dataset, splits, generator=generator)

    if train:
        return dataset_train

    return dataset_val
