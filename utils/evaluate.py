import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)
import json
import os


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for X, y in loader:
        X = X.to(device)
        preds = model(X).argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def evaluate(model, loader, device, le, model_name="model", save_dir="runs"):
    preds, targets = get_predictions(model, loader, device)

    acc          = (preds == targets).mean()
    bal_acc      = balanced_accuracy_score(targets, preds)
    macro_f1     = f1_score(targets, preds, average="macro")

    print(f"\n{'='*50}")
    print(f"Results for: {model_name}")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Macro F1         : {macro_f1:.4f}")
    print(f"{'='*50}")
    print("\nPer-class report:")
    print(classification_report(targets, preds, target_names=le.classes_))

    results = {
        "model": model_name,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{model_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results
