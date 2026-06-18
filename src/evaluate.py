"""Evaluate the trained CLIP + MLP classifier on the held-out test split.

Loads the best checkpoint, runs inference on the test split, and reports
accuracy / precision / recall / F1 / confusion matrix. Also dumps a sample of
misclassified examples (image path + caption + predicted vs actual) to a CSV
for manual inspection.

Example (offline smoke test, after training with --dummy)::

    python src/evaluate.py --dummy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# Support both `python src/evaluate.py` and `python -m src.evaluate`.
try:
    from .dataset import (
        LABEL_FALSIFIED,
        LABEL_PRISTINE,
        build_dataloaders,
        generate_dummy_dataset,
    )
    from .model import CLIPMLPClassifier
except ImportError:  # pragma: no cover - direct-script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset import (
        LABEL_FALSIFIED,
        LABEL_PRISTINE,
        build_dataloaders,
        generate_dummy_dataset,
    )
    from model import CLIPMLPClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "misclassified.csv"

LABEL_NAMES = {LABEL_PRISTINE: "pristine", LABEL_FALSIFIED: "falsified"}


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


def load_model(checkpoint_path: Path, device: torch.device) -> CLIPMLPClassifier:
    """Rebuild the model from the checkpoint's stored config and load weights."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model = CLIPMLPClassifier(
        clip_name=config.get("clip_model", "openai/clip-vit-base-patch32"),
        hidden_dims=tuple(config.get("hidden_dims", (512, 256))),
        num_classes=1,  # single-logit head (BCE), matching training
        dropout=config.get("dropout", 0.3),
        freeze_clip=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss")
    print(
        f"[eval] loaded checkpoint '{checkpoint_path.name}' "
        f"(epoch {epoch}"
        + (f", val_loss {val_loss:.4f}" if val_loss is not None else "")
        + ")"
    )
    return model


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


@torch.no_grad()
def run_inference(
    model: CLIPMLPClassifier,
    loader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Run the model over ``loader`` and collect predictions + metadata.

    Returns a dict with parallel lists: image_paths, captions, y_true, y_pred,
    and probabilities (of the falsified/positive class).
    """
    image_paths: list[str] = []
    captions: list[str] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    probs: list[float] = []

    for batch_paths, batch_caps, labels in tqdm(loader, desc="[eval] inference"):
        logits = model(images=batch_paths, texts=batch_caps).squeeze(-1)
        batch_probs = torch.sigmoid(logits)
        preds = (batch_probs > threshold).long()

        image_paths.extend(batch_paths)
        captions.extend(batch_caps)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
        probs.extend(batch_probs.cpu().tolist())

    return {
        "image_paths": image_paths,
        "captions": captions,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def report_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute and print accuracy / precision / recall / F1 + confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    # Positive class = falsified (1).
    precision = precision_score(y_true, y_pred, pos_label=LABEL_FALSIFIED, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=LABEL_FALSIFIED, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=LABEL_FALSIFIED, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[LABEL_PRISTINE, LABEL_FALSIFIED])

    print("\n" + "=" * 48)
    print(f"  samples   : {len(y_true)}")
    print(f"  accuracy  : {acc:.4f}")
    print(f"  precision : {precision:.4f}   (positive = falsified)")
    print(f"  recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print("=" * 48)
    print("  confusion matrix (rows = actual, cols = predicted):")
    print(f"                  pred:pristine  pred:falsified")
    print(f"  actual:pristine    {cm[0, 0]:>10d}    {cm[0, 1]:>12d}")
    print(f"  actual:falsified   {cm[1, 0]:>10d}    {cm[1, 1]:>12d}")
    print("=" * 48 + "\n")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def save_misclassified(results: dict, output_path: Path, n: int = 10) -> int:
    """Write up to ``n`` misclassified examples to a CSV. Returns count written."""
    rows = []
    for path, cap, true, pred, prob in zip(
        results["image_paths"],
        results["captions"],
        results["y_true"],
        results["y_pred"],
        results["probs"],
    ):
        if pred != true:
            rows.append(
                {
                    "image_path": path,
                    "caption": cap,
                    "actual_label": true,
                    "actual_name": LABEL_NAMES[true],
                    "predicted_label": pred,
                    "predicted_name": LABEL_NAMES[pred],
                    "prob_falsified": round(prob, 4),
                }
            )
            if len(rows) >= n:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=[
            "image_path",
            "caption",
            "actual_label",
            "actual_name",
            "predicted_label",
            "predicted_name",
            "prob_falsified",
        ],
    ).to_csv(output_path, index=False)
    print(f"[eval] saved {len(rows)} misclassified example(s) to {output_path}")
    return len(rows)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[eval] device={device}")

    data_root = Path(args.data_root)
    if args.dummy:
        data_root = data_root / "_dummy"
        # Regenerate only if the test split isn't already present.
        test_json = data_root / "news_clippings" / "data" / args.subset / "test.json"
        if not test_json.exists():
            generate_dummy_dataset(data_root, subset=args.subset, seed=args.seed)

    loader = build_dataloaders(
        data_root=data_root,
        subset=args.subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        splits=("test",),
    )["test"]

    model = load_model(Path(args.checkpoint), device)
    results = run_inference(model, loader, device, threshold=args.threshold)
    report_metrics(results["y_true"], results["y_pred"])
    save_misclassified(results, Path(args.output), n=args.num_misclassified)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the CLIP+MLP fake-news classifier.")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--data-root", default="data")
    p.add_argument("--subset", default="merged_balanced")
    p.add_argument("--dummy", action="store_true",
                   help="Evaluate on synthetic data (offline smoke test).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability threshold for the falsified class.")
    p.add_argument("--device", default="", help="cuda / cpu (auto if empty).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="CSV path for misclassified examples.")
    p.add_argument("--num-misclassified", type=int, default=10)
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
