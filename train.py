"""
Main script — trains PatchTST, UniTS, and the InceptionTime baseline on LSST.

Usage:
    python train.py --model patchtst    # train PatchTST only
    python train.py --model units       # train UniTS only
    python train.py --model baseline    # train baseline only
    python train.py --model all         # train all three
    python train.py --model both        # train PatchTST + baseline (legacy)
"""

import argparse
import torch
import random
import numpy as np

from data.dataset      import get_dataloaders
from models.patchtst   import build_model
from models.units      import build_units
from models.baseline   import build_baseline
from utils.trainer     import train
from utils.evaluate    import evaluate


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader, n_classes, seq_len, n_channels, le = get_dataloaders(
        batch_size=args.batch_size
    )
    print(f"seq_len={seq_len}, n_channels={n_channels}, n_classes={n_classes}")

    results = {}

    # ── PatchTST ──────────────────────────────────────────────────
    if args.model in ("patchtst", "both", "all"):
        print("\n" + "="*50)
        print("Training PatchTST")
        print("="*50)
        model = build_model(seq_len, n_channels, n_classes, device)
        history, best_acc = train(
            model, train_loader, test_loader,
            model_name="patchtst",
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            save_dir="runs",
            device=device,
        )
        results["patchtst"] = evaluate(model, test_loader, device, le,
                                        model_name="patchtst", save_dir="runs")

    # ── UniTS ─────────────────────────────────────────────────────
    if args.model in ("units", "all"):
        print("\n" + "="*50)
        print("Training UniTS (NeurIPS 2024)")
        print("="*50)
        units = build_units(seq_len, n_channels, n_classes, device)
        history, best_acc = train(
            units, train_loader, test_loader,
            model_name="units",
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            save_dir="runs",
            device=device,
        )
        results["units"] = evaluate(units, test_loader, device, le,
                                     model_name="units", save_dir="runs")

    # ── Baseline ──────────────────────────────────────────────────
    if args.model in ("baseline", "both", "all"):
        print("\n" + "="*50)
        print("Training InceptionTime baseline")
        print("="*50)
        baseline = build_baseline(n_channels, n_classes, device)
        history, best_acc = train(
            baseline, train_loader, test_loader,
            model_name="baseline",
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            save_dir="runs",
            device=device,
        )
        results["baseline"] = evaluate(baseline, test_loader, device, le,
                                         model_name="baseline", save_dir="runs")

    # ── Summary ───────────────────────────────────────────────────
    if len(results) > 1:
        print("\n" + "="*50)
        print("FINAL COMPARISON")
        print("="*50)
        for name, r in results.items():
            print(f"  {name:12s} | acc={r['accuracy']:.4f} | "
                  f"bal_acc={r['balanced_accuracy']:.4f} | "
                  f"f1={r['macro_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str,   default="all",
                        choices=["patchtst", "units", "baseline", "both", "all"])
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=15)
    args = parser.parse_args()
    main(args)
