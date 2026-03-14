import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import LabelEncoder


def load_lsst():
    """Load LSST dataset from UEA archive."""
    print("Loading LSST dataset...")
    ds = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ds.load_dataset("LSST")

    # Encode labels to integers
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Classes: {le.classes_}")
    print(f"  Timesteps: {X_train.shape[1]}  |  Channels: {X_train.shape[2]}")

    return X_train, y_train_enc, X_test, y_test_enc, le


def normalize(X_train, X_test):
    """Normalize per channel using train statistics."""
    mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
    std  = np.nanstd(X_train,  axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std
    # Replace NaN with 0 (LSST has missing values)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0)
    return X_train, X_test


class LSSTPatchTSTDataset(Dataset):
    """
    Dataset for PatchTST.
    Input shape expected: (batch, seq_len, n_channels)
    """
    def __init__(self, X, y):
        # X: (N, T, C) as numpy → keep as (N, T, C)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(batch_size=64):
    X_train, y_train, X_test, y_test, le = load_lsst()
    X_train, X_test = normalize(X_train, X_test)

    train_ds = LSSTPatchTSTDataset(X_train, y_train)
    test_ds  = LSSTPatchTSTDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    n_classes  = len(le.classes_)
    seq_len    = X_train.shape[1]
    n_channels = X_train.shape[2]

    return train_loader, test_loader, n_classes, seq_len, n_channels, le
