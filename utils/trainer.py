import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import time
import os
import json


class EarlyStopping:
    def __init__(self, patience=15, delta=1e-4):
        self.patience  = patience
        self.delta     = delta
        self.counter   = 0
        self.best_loss = np.inf
        self.stop      = False

    def __call__(self, val_loss, model, path):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0., 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0., 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)

    return total_loss / n, correct / n


def train(
    model,
    train_loader,
    test_loader,
    model_name="model",
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    patience=15,
    save_dir="runs",
    device="cpu",
):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"{model_name}_best.pt")

    criterion   = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer   = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler   = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    early_stop  = EarlyStopping(patience=patience)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = eval_epoch(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_acc:
            best_acc = vl_acc

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"Val loss {vl_loss:.4f} acc {vl_acc:.4f} | "
                f"Best acc {best_acc:.4f} | "
                f"{elapsed:.0f}s"
            )

        early_stop(vl_loss, model, ckpt_path)
        if early_stop.stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Save history
    with open(os.path.join(save_dir, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val accuracy: {best_acc:.4f}")
    return history, best_acc
