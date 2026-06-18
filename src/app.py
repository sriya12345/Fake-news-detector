"""Gradio demo for the CLIP + MLP fake-news detector.

A user pastes a news headline and uploads an image. The app loads the trained
model, runs inference, and shows:

    1. a consistency score (0-100; higher = text and image agree),
    2. a verdict badge (likely real / likely manipulated),
    3. a cosine-similarity heatmap of the caption against image patches,
       i.e. *where* in the image the text matches.

Launch::

    python src/app.py                       # uses checkpoints/best_model.pt
    python src/app.py --checkpoint path.pt --share
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to buffer, never to a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Support both `python src/app.py` and `python -m src.app`.
try:
    from .model import CLIPMLPClassifier
except ImportError:  # pragma: no cover - direct-script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import CLIPMLPClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"

# CLIP preprocessing target; base-patch32 -> 224/32 = 7x7 patch grid.
CLIP_INPUT_SIZE = 224

# Module-level handles set by load_model().
_MODEL: CLIPMLPClassifier | None = None
_DEVICE: torch.device | None = None
_TRAINED: bool = False


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


def load_model(checkpoint_path: Path, device: torch.device) -> None:
    """Load the trained model into module globals (once, at startup)."""
    global _MODEL, _DEVICE, _TRAINED
    _DEVICE = device

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        model = CLIPMLPClassifier(
            clip_name=config.get("clip_model", "openai/clip-vit-base-patch32"),
            hidden_dims=tuple(config.get("hidden_dims", (512, 256))),
            num_classes=1,
            dropout=config.get("dropout", 0.3),
            freeze_clip=True,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        _TRAINED = True
        print(f"[app] loaded trained checkpoint: {checkpoint_path}")
    else:
        # Still let the demo run, but flag that the head is untrained.
        model = CLIPMLPClassifier(num_classes=1, freeze_clip=True).to(device)
        _TRAINED = False
        print(f"[app] WARNING: no checkpoint at {checkpoint_path}; "
              "using an UNTRAINED head (verdict is meaningless).")

    model.eval()
    _MODEL = model


# --------------------------------------------------------------------------- #
# Heatmap: caption vs image patches
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _patch_similarity(image: Image.Image, text: str) -> np.ndarray:
    """Cosine similarity between the caption and each image patch.

    Projects every vision patch token into CLIP's joint embedding space (the
    same projection used for the pooled image embedding), L2-normalizes, and
    dots against the normalized text embedding. Returns a ``(grid, grid)`` array.
    """
    assert _MODEL is not None and _DEVICE is not None
    clip = _MODEL.clip

    inputs = _MODEL.processor(
        text=[text], images=[image], return_tensors="pt", padding=True, truncation=True
    )
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

    # Per-patch vision features -> joint space.
    vision_out = clip.vision_model(pixel_values=inputs["pixel_values"])
    patch_tokens = vision_out.last_hidden_state[:, 1:, :]  # drop CLS
    patch_tokens = clip.vision_model.post_layernorm(patch_tokens)
    patch_emb = clip.visual_projection(patch_tokens)  # (1, n_patches, dim)
    patch_emb = F.normalize(patch_emb, dim=-1)

    # Text embedding -> joint space.
    text_out = clip.get_text_features(
        input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
    )
    text_emb = F.normalize(_MODEL._as_embedding(text_out), dim=-1)  # (1, dim)

    sim = (patch_emb * text_emb.unsqueeze(1)).sum(-1).squeeze(0)  # (n_patches,)
    grid = int(round(sim.numel() ** 0.5))
    return sim.reshape(grid, grid).cpu().numpy()


def _render_heatmap(image: Image.Image, sim_grid: np.ndarray) -> Image.Image:
    """Overlay the patch-similarity grid on the (resized) image as a heatmap."""
    base = image.convert("RGB").resize((CLIP_INPUT_SIZE, CLIP_INPUT_SIZE))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(base)
    hm = ax.imshow(
        sim_grid,
        extent=(0, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE, 0),
        cmap="jet",
        alpha=0.5,
        interpolation="bilinear",
    )
    ax.axis("off")
    ax.set_title("Caption vs image-patch similarity")
    fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


def _verdict_badge(score: float, prob_manip: float, threshold: float) -> str:
    """Build a colored HTML badge for the verdict + score."""
    manipulated = prob_manip > threshold
    label = "LIKELY MANIPULATED" if manipulated else "LIKELY REAL"
    color = "#dc2626" if manipulated else "#16a34a"  # red / green
    note = (
        "" if _TRAINED
        else "<div style='color:#b45309;font-size:13px;margin-top:8px;'>"
             "⚠ No trained checkpoint loaded — verdict is not meaningful.</div>"
    )
    return (
        f"<div style='text-align:center;padding:18px;border-radius:12px;"
        f"background:{color};color:white;'>"
        f"<div style='font-size:26px;font-weight:800;'>{label}</div>"
        f"<div style='font-size:15px;margin-top:6px;'>"
        f"Consistency score: {score:.0f}/100</div></div>{note}"
    )


@torch.no_grad()
def analyze(headline: str, image: Image.Image | None, threshold: float = 0.5):
    """Gradio callback: returns (score, verdict_html, heatmap_image)."""
    if not headline or not headline.strip():
        return 0.0, "<div style='padding:12px;'>Please enter a headline.</div>", None
    if image is None:
        return 0.0, "<div style='padding:12px;'>Please upload an image.</div>", None

    assert _MODEL is not None
    image_emb, text_emb = _MODEL.encode([image], [headline])
    logit = _MODEL(image_emb=image_emb, text_emb=text_emb).squeeze(-1)
    prob_manip = torch.sigmoid(logit).item()  # P(falsified)

    consistency = (1.0 - prob_manip) * 100.0
    badge = _verdict_badge(consistency, prob_manip, threshold)
    heatmap = _render_heatmap(image, _patch_similarity(image, headline))
    return round(consistency, 1), badge, heatmap


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #


def build_demo(threshold: float = 0.5):
    import gradio as gr

    with gr.Blocks(title="Fake News Detector") as demo:
        gr.Markdown(
            "# 📰 Multimodal Fake News Detector\n"
            "Paste a news headline and upload its image. The model checks "
            "whether the text and image are *semantically consistent* using CLIP."
        )
        with gr.Row():
            with gr.Column():
                headline = gr.Textbox(
                    label="News headline / caption",
                    placeholder="e.g. Protesters gather outside the capitol...",
                    lines=3,
                )
                image = gr.Image(label="Image", type="pil")
                analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                verdict = gr.HTML(label="Verdict")
                score = gr.Number(label="Consistency score (0–100)", precision=1)
                heatmap = gr.Image(label="Text–image similarity heatmap")

        analyze_btn.click(
            fn=lambda h, im: analyze(h, im, threshold),
            inputs=[headline, image],
            outputs=[score, verdict, heatmap],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the fake-news Gradio demo.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    load_model(Path(args.checkpoint), device)
    demo = build_demo(args.threshold)
    demo.launch(server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
