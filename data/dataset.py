import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import LabelEncoder


def load_lsst():
    print("Loading LSST dataset...")
    ds = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ds.load_dataset("LSST")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Classes: {le.classes_}")
    print(f"  Timesteps: {X_train.shape[1]}  |  Channels: {X_train.shape[2]}")

    return X_train, y_train_enc, X_test, y_test_enc, le


def normalize(X_train, X_test):
    mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
    std  = np.nanstd(X_train,  axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0)
    return X_train, X_test


class LSSTPatchTSTDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()   # (T, C)

        if self.augment:
            # 1. Jitter — add small gaussian noise
            if torch.rand(1) < 0.5:
                x = x + torch.randn_like(x) * 0.05

            # 2. Scaling — multiply by a random scalar per channel
            if torch.rand(1) < 0.5:
                scale = 0.9 + torch.rand(x.shape[-1]) * 0.2   # [0.9, 1.1]
                x = x * scale.unsqueeze(0)

            # 3. Time shift — roll the series by a few steps
            if torch.rand(1) < 0.3:
                shift = torch.randint(-3, 4, (1,)).item()
                x = torch.roll(x, shift, dims=0)

        return x, self.y[idx]


def get_dataloaders(batch_size=64):
    X_train, y_train, X_test, y_test, le = load_lsst()
    X_train, X_test = normalize(X_train, X_test)

    # augment=True only on train, never on test
    train_ds = LSSTPatchTSTDataset(X_train, y_train, augment=True)
    test_ds  = LSSTPatchTSTDataset(X_test,  y_test,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    n_classes  = len(le.classes_)
    seq_len    = X_train.shape[1]
    n_channels = X_train.shape[2]

    return train_loader, test_loader, n_classes, seq_len, n_channels, le