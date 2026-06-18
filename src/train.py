"""Training loop for the CLIP + MLP fake-news classifier.

Trains the MLP head from :mod:`model` (CLIP stays frozen) with:

    * binary cross-entropy loss (BCEWithLogitsLoss, single-logit head)
    * Adam optimizer
    * early stopping on validation loss
    * Weights & Biases logging of loss and accuracy
    * best checkpoint saved to ``checkpoints/best_model.pt``

Run a quick end-to-end smoke test on synthetic data::

    python src/train.py --dummy --epochs 3 --wandb-mode disabled
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Support both `python src/train.py` and `python -m src.train`.
try:
    from .dataset import build_dataloaders, generate_dummy_dataset
    from .model import CLIPMLPClassifier
except ImportError:  # pragma: no cover - direct-script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset import build_dataloaders, generate_dummy_dataset
    from model import CLIPMLPClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _init_wandb(args: argparse.Namespace):
    """Initialize a W&B run, or return None if W&B is unavailable.

    Honors ``--wandb-mode`` (online / offline / disabled). Missing package is
    not fatal: we warn and fall back to console-only logging.
    """
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed - logging to console only. "
              "`uv pip install wandb` to enable.")
        return None

    return wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        config=vars(args),
    )


def _log(run, metrics: dict, step: int) -> None:
    if run is not None:
        run.log(metrics, step=step)


# --------------------------------------------------------------------------- #
# Train / eval one epoch
# --------------------------------------------------------------------------- #


def run_epoch(
    model: CLIPMLPClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
) -> tuple[float, float]:
    """Run one pass over ``loader``. Trains if ``optimizer`` is given, else evals.

    Returns ``(avg_loss, accuracy)``.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for image_paths, captions, labels in tqdm(loader, desc=desc, leave=False):
            targets = labels.float().to(device)  # BCE wants float targets

            logits = model(images=image_paths, texts=captions).squeeze(-1)
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * targets.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == labels.to(device)).sum().item()
            total += targets.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


# --------------------------------------------------------------------------- #
# Training driver
# --------------------------------------------------------------------------- #


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[train] device={device}")

    # Data --------------------------------------------------------------- #
    data_root = Path(args.data_root)
    if args.dummy:
        data_root = data_root / "_dummy"
        generate_dummy_dataset(data_root, subset=args.subset, seed=args.seed)

    loaders = build_dataloaders(
        data_root=data_root,
        subset=args.subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        splits=("train", "val"),
    )

    # Model -------------------------------------------------------------- #
    # Single-logit head so we can use binary cross-entropy directly.
    model = CLIPMLPClassifier(
        clip_name=args.clip_model,
        hidden_dims=tuple(args.hidden_dims),
        num_classes=1,
        dropout=args.dropout,
        freeze_clip=True,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    # Only head params require grad (CLIP is frozen).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    print(f"[train] trainable params: {sum(p.numel() for p in trainable):,}")

    run = _init_wandb(args)

    # Training loop with early stopping ---------------------------------- #
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, device, optimizer,
            desc=f"epoch {epoch} [train]",
        )
        val_loss, val_acc = run_epoch(
            model, loaders["val"], criterion, device, optimizer=None,
            desc=f"epoch {epoch} [val]",
        )

        print(
            f"epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )
        _log(
            run,
            {
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "epoch": epoch,
            },
            step=epoch,
        )

        # Checkpoint + early stopping on val loss.
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": vars(args),
                },
                checkpoint_path,
            )
            print(f"  -> new best val loss {val_loss:.4f}; saved {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(
                    f"[early stop] no val improvement for {args.patience} "
                    f"epochs (best={best_val_loss:.4f})."
                )
                break

    print(f"[train] done. best val loss={best_val_loss:.4f}")
    if run is not None:
        run.summary["best_val_loss"] = best_val_loss
        run.finish()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the CLIP+MLP fake-news classifier.")
    # Data
    p.add_argument("--data-root", default="data", help="Dataset root directory.")
    p.add_argument("--subset", default="merged_balanced", help="NewsCLIPpings subset.")
    p.add_argument("--dummy", action="store_true",
                   help="Generate + train on synthetic data (offline smoke test).")
    p.add_argument("--num-workers", type=int, default=0)
    # Model
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256])
    p.add_argument("--dropout", type=float, default=0.3)
    # Optimization
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5,
                   help="Early-stopping patience (epochs without val improvement).")
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="Minimum val-loss decrease to count as improvement.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="", help="cuda / cpu (auto if empty).")
    # Checkpoint / logging
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--wandb-project", default="fake-news-detector")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"])
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
