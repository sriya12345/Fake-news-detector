"""CLIP-based multimodal classifier for out-of-context misinformation.

A frozen CLIP backbone (``openai/clip-vit-base-patch32``) extracts L2-normalized
image and text embeddings. An MLP head is trained on top of the feature vector::

    [ image_emb (512) , text_emb (512) , cosine_similarity (1) ]  -> 1025 dims

and predicts a binary label.

Label convention (matches ``src/dataset.py``)::

    0 = real / pristine      (image and caption genuinely belong together)
    1 = manipulated / falsified (out-of-context mismatch)

CLIP weights are frozen, so only the MLP head is trained.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# `transformers` is imported lazily inside CLIPMLPClassifier so the MLP head
# can be used/tested without pulling in the (heavy) CLIP dependency.

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

# Same convention as dataset.py.
LABEL_PRISTINE = 0
LABEL_FALSIFIED = 1

ImageInput = str | Path | Image.Image


# --------------------------------------------------------------------------- #
# MLP head
# --------------------------------------------------------------------------- #


class MLPHead(nn.Module):
    """Simple MLP: stacked Linear -> ReLU -> Dropout blocks, then a logit layer."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] = (512, 256),
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            dim = hidden
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# CLIP + MLP classifier
# --------------------------------------------------------------------------- #


class CLIPMLPClassifier(nn.Module):
    """Frozen CLIP feature extractor + trainable MLP classifier head.

    Parameters
    ----------
    clip_name:
        HuggingFace CLIP checkpoint to load.
    hidden_dims:
        Widths of the MLP hidden layers.
    num_classes:
        Output classes (2 for the real/manipulated task).
    dropout:
        Dropout probability in the MLP.
    freeze_clip:
        If True (default) CLIP params are frozen and run under ``no_grad``.
    """

    def __init__(
        self,
        clip_name: str = DEFAULT_CLIP_MODEL,
        hidden_dims: Sequence[int] = (512, 256),
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_clip: bool = True,
    ) -> None:
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.clip = CLIPModel.from_pretrained(clip_name)
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        self.freeze_clip = freeze_clip

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad_(False)
            self.clip.eval()

        self.embed_dim: int = self.clip.config.projection_dim  # 512 for base-patch32
        # Feature vector: [image_emb, text_emb, cosine_similarity].
        self.feature_dim = self.embed_dim * 2 + 1
        self.head = MLPHead(self.feature_dim, hidden_dims, num_classes, dropout)

    # -- utilities --------------------------------------------------------- #

    @property
    def device(self) -> torch.device:
        return next(self.head.parameters()).device

    def train(self, mode: bool = True):  # type: ignore[override]
        """Keep CLIP in eval mode when frozen (so dropout/BN don't drift)."""
        super().train(mode)
        if self.freeze_clip:
            self.clip.eval()
        return self

    @staticmethod
    def _load_image(image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    # -- encoding ---------------------------------------------------------- #

    def encode(
        self,
        images: Sequence[ImageInput],
        texts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images and texts to L2-normalized CLIP embeddings.

        ``images`` may be file paths or PIL images. Returns
        ``(image_emb, text_emb)``, each shaped ``(batch, embed_dim)``.
        """
        pil_images = [self._load_image(im) for im in images]
        inputs = self.processor(
            text=list(texts),
            images=pil_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        ctx = torch.no_grad() if self.freeze_clip else nullcontext()
        with ctx:
            image_out = self.clip.get_image_features(pixel_values=inputs["pixel_values"])
            text_out = self.clip.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )

        image_emb = F.normalize(self._as_embedding(image_out), dim=-1)
        text_emb = F.normalize(self._as_embedding(text_out), dim=-1)
        return image_emb, text_emb

    @staticmethod
    def _as_embedding(out) -> torch.Tensor:
        """Extract the embedding tensor across transformers versions.

        transformers <5 returns the projected tensor directly; >=5 wraps it in a
        ``BaseModelOutputWithPooling`` whose ``pooler_output`` is the projection.
        """
        if isinstance(out, torch.Tensor):
            return out
        return out.pooler_output

    def build_features(
        self, image_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate ``[image_emb, text_emb, cosine_similarity]``.

        Embeddings are assumed L2-normalized, so the dot product is the cosine
        similarity.
        """
        cosine = (image_emb * text_emb).sum(dim=-1, keepdim=True)
        return torch.cat([image_emb, text_emb, cosine], dim=-1)

    # -- forward ----------------------------------------------------------- #

    def forward(
        self,
        images: Sequence[ImageInput] | None = None,
        texts: Sequence[str] | None = None,
        *,
        image_emb: torch.Tensor | None = None,
        text_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return class logits of shape ``(batch, num_classes)``.

        Either pass raw ``images``/``texts`` (CLIP encodes them) or pass
        precomputed ``image_emb``/``text_emb`` (handy for caching CLIP outputs).
        """
        if image_emb is None or text_emb is None:
            if images is None or texts is None:
                raise ValueError(
                    "Provide either (images, texts) or (image_emb, text_emb)."
                )
            image_emb, text_emb = self.encode(images, texts)

        features = self.build_features(image_emb, text_emb)
        return self.head(features)

    @torch.no_grad()
    def predict(
        self, images: Sequence[ImageInput], texts: Sequence[str]
    ) -> torch.Tensor:
        """Convenience inference: return predicted class indices."""
        self.eval()
        logits = self.forward(images, texts)
        return logits.argmax(dim=-1)


# --------------------------------------------------------------------------- #
# Self-test (no network: exercises the MLP head only)
# --------------------------------------------------------------------------- #


def _selftest_head() -> None:
    """Verify the head shapes with random normalized embeddings (no CLIP download)."""
    torch.manual_seed(0)
    embed_dim = 512
    batch = 4
    head = MLPHead(embed_dim * 2 + 1)

    image_emb = F.normalize(torch.randn(batch, embed_dim), dim=-1)
    text_emb = F.normalize(torch.randn(batch, embed_dim), dim=-1)
    cosine = (image_emb * text_emb).sum(dim=-1, keepdim=True)
    features = torch.cat([image_emb, text_emb, cosine], dim=-1)

    logits = head(features)
    print(f"[selftest] feature_dim={features.shape[-1]} logits={tuple(logits.shape)}")
    assert features.shape == (batch, embed_dim * 2 + 1)
    assert logits.shape == (batch, 2)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[selftest] MLP head params: {n_params:,}")
    print("[selftest] OK")


if __name__ == "__main__":
    _selftest_head()
