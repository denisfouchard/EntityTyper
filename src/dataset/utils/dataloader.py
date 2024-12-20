from torch.utils.data import Dataset, DataLoader
from typing import Callable


def dataset_loader(
    dataset: Dataset,
    args,
    collate_fn: Callable,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    A simple data loader for PyTorch datasets.
    """
    idx_split = dataset.get_idx_split()
    train_dataset = [dataset[i] for i in idx_split["train"]]
    val_dataset = [dataset[i] for i in idx_split["val"]]
    test_dataset = [dataset[i] for i in idx_split["test"]]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
