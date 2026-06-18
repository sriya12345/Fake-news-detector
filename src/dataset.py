"""NewsCLIPpings dataset download and loading.

NewsCLIPpings (Luo, Darrell & Rohrbach, EMNLP 2021) is a benchmark for
out-of-context image-caption misinformation. It is built *on top of* the
VisualNews corpus and therefore does not ship its own images. Instead, each
annotation references:

    * an ``id``       -> a caption in VisualNews
    * an ``image_id`` -> an image  in VisualNews   (possibly a *different* article)
    * a ``falsified`` flag -> whether the (image, caption) pair is mismatched

So building one training example means a double lookup into VisualNews:
the caption text comes from ``id`` and the image path comes from ``image_id``.

Expected on-disk layout (the standard one from the official releases)::

    <data_root>/
    ├── visual_news/
    │   └── origin/
    │       ├── data.json                 # list of VisualNews articles
    │       └── <source>/images/...       # the actual .jpg files
    └── news_clippings/
        └── data/
            └── <split>/                  # e.g. merged_balanced
                ├── train.json
                ├── val.json
                └── test.json

Access note
-----------
The NewsCLIPpings annotations are public (Google Drive / GitHub), but the
VisualNews images are gated behind a request form
(https://github.com/FuxiaoLiu/VisualNews-Repository). Full unattended
auto-download is therefore not possible; :func:`download_newsclippings`
fetches what it can and prints instructions for the gated part.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Paths & config
# --------------------------------------------------------------------------- #

# Repo root = parent of this file's parent (src/ -> project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

Split = Literal["train", "val", "test"]

# Label convention: a *falsified* (out-of-context) pair is the positive class.
LABEL_PRISTINE = 0  # image and caption genuinely belong together
LABEL_FALSIFIED = 1  # image and caption mismatched -> "fake"

# The NewsCLIPpings paper defines several subsets; "merged_balanced" is the
# main balanced benchmark and a sensible default.
DEFAULT_SUBSET = "merged_balanced"


@dataclass
class Sample:
    """One example: where the image lives, its caption, and the label."""

    image_path: str
    caption_text: str
    label: int


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def download_newsclippings(data_root: Path | str = DEFAULT_DATA_ROOT) -> Path:
    """Ensure the dataset is present under ``data_root``.

    Annotations can be cloned automatically; VisualNews images are gated and
    must be requested manually. This function verifies what is present and
    prints actionable instructions for anything missing, then returns the
    resolved ``data_root``.
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    visualnews_data = data_root / "visual_news" / "origin" / "data.json"
    clippings_dir = data_root / "news_clippings" / "data"

    missing: list[str] = []

    if not clippings_dir.exists():
        missing.append(
            "NewsCLIPpings annotations not found.\n"
            f"    Expected: {clippings_dir}\n"
            "    Get them from: https://github.com/g-luo/news_clippings\n"
            "    (download the `news_clippings` annotation folder and place it "
            "under <data_root>/news_clippings/)."
        )

    if not visualnews_data.exists():
        missing.append(
            "VisualNews corpus not found (images + data.json).\n"
            f"    Expected: {visualnews_data}\n"
            "    VisualNews is access-gated; request it here:\n"
            "    https://github.com/FuxiaoLiu/VisualNews-Repository\n"
            "    then extract it to <data_root>/visual_news/origin/."
        )

    if missing:
        print("=" * 72)
        print("NewsCLIPpings is not fully downloaded yet. Action needed:\n")
        for i, msg in enumerate(missing, 1):
            print(f"  {i}. {msg}\n")
        print("=" * 72)
    else:
        print(f"NewsCLIPpings looks ready under: {data_root}")

    return data_root


# --------------------------------------------------------------------------- #
# VisualNews index
# --------------------------------------------------------------------------- #


def _load_visualnews_index(visualnews_root: Path) -> dict[int, dict]:
    """Map VisualNews ``id`` -> {"image_path", "caption"} for fast lookup.

    ``visualnews_root`` is the directory containing ``data.json`` (i.e.
    ``<data_root>/visual_news/origin``). Image paths in VisualNews are stored
    relative to this directory, so we resolve them to absolute paths here.
    """
    data_json = visualnews_root / "data.json"
    if not data_json.exists():
        raise FileNotFoundError(
            f"VisualNews data.json not found at {data_json}. "
            "Run download_newsclippings() for setup instructions."
        )

    with open(data_json, "r", encoding="utf-8") as f:
        articles = json.load(f)

    index: dict[int, dict] = {}
    for art in articles:
        rel = art["image_path"].lstrip("./")
        index[int(art["id"])] = {
            "image_path": str((visualnews_root / rel).resolve()),
            "caption": art["caption"],
        }
    return index


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class NewsCLIPpingsDataset(Dataset):
    """PyTorch Dataset yielding ``(image_path, caption_text, label)`` tuples.

    Parameters
    ----------
    data_root:
        Root directory holding ``visual_news/`` and ``news_clippings/``.
    split:
        One of ``"train"``, ``"val"``, ``"test"``.
    subset:
        NewsCLIPpings subset name (default ``"merged_balanced"``).
    """

    def __init__(
        self,
        data_root: Path | str = DEFAULT_DATA_ROOT,
        split: Split = "train",
        subset: str = DEFAULT_SUBSET,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.subset = subset

        visualnews_root = self.data_root / "visual_news" / "origin"
        self._vn_index = _load_visualnews_index(visualnews_root)

        ann_file = self.data_root / "news_clippings" / "data" / subset / f"{split}.json"
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}. "
                "Run download_newsclippings() for setup instructions."
            )
        with open(ann_file, "r", encoding="utf-8") as f:
            self.annotations: list[dict] = json.load(f)["annotations"]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[str, str, int]:
        ann = self.annotations[idx]
        # Caption comes from `id`, image from `image_id` (the NewsCLIPpings twist).
        caption_text = self._vn_index[int(ann["id"])]["caption"]
        image_path = self._vn_index[int(ann["image_id"])]["image_path"]
        label = LABEL_FALSIFIED if ann["falsified"] else LABEL_PRISTINE
        return image_path, caption_text, label


def _collate(
    batch: list[tuple[str, str, int]],
) -> tuple[tuple[str, ...], tuple[str, ...], torch.Tensor]:
    """Keep paths/captions as tuples of strings; stack labels into a tensor."""
    image_paths, captions, labels = zip(*batch)
    return image_paths, captions, torch.tensor(labels, dtype=torch.long)


# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #


def build_dataloaders(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    subset: str = DEFAULT_SUBSET,
    batch_size: int = 32,
    num_workers: int = 0,
    splits: tuple[Split, ...] = ("train", "val", "test"),
) -> dict[str, DataLoader]:
    """Build a DataLoader per split.

    Each batch is ``(image_paths, caption_texts, labels)`` where the first two
    are tuples of strings and ``labels`` is a ``LongTensor``. Returns a dict
    keyed by split name. Only ``train`` is shuffled.
    """
    loaders: dict[str, DataLoader] = {}
    for split in splits:
        ds = NewsCLIPpingsDataset(data_root, split=split, subset=subset)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=_collate,
        )
    return loaders


# --------------------------------------------------------------------------- #
# Dummy data (for offline prototyping)
# --------------------------------------------------------------------------- #


def generate_dummy_dataset(
    data_root: Path | str,
    n_articles: int = 20,
    subset: str = DEFAULT_SUBSET,
    image_size: tuple[int, int] = (64, 64),
    seed: int = 0,
) -> Path:
    """Write a tiny synthetic dataset in the real on-disk layout.

    Useful for exercising the loader / sanity check without the access-gated
    VisualNews download. It fabricates a VisualNews ``data.json`` + solid-color
    .jpg images, then NewsCLIPpings ``{train,val,test}.json`` annotations.

    Pristine samples pair a caption with its own image; falsified samples pair
    it with a *different* article's image (mimicking the real benchmark).
    Returns the resolved ``data_root``.
    """
    import random

    from PIL import Image

    rng = random.Random(seed)
    data_root = Path(data_root)

    # --- VisualNews articles + images -------------------------------------- #
    vn_root = data_root / "visual_news" / "origin"
    sources = ["the_guardian", "washington_post", "bbc"]
    articles = []
    for i in range(n_articles):
        source = sources[i % len(sources)]
        rel_path = f"{source}/images/{i:04d}.jpg"
        img_abs = vn_root / rel_path
        img_abs.parent.mkdir(parents=True, exist_ok=True)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        Image.new("RGB", image_size, color).save(img_abs, "JPEG")
        articles.append(
            {
                "id": i,
                "image_id": i,
                "image_path": f"./{rel_path}",
                "caption": f"Dummy caption #{i} from {source.replace('_', ' ')}.",
                "source": source,
            }
        )
    vn_root.mkdir(parents=True, exist_ok=True)
    with open(vn_root / "data.json", "w", encoding="utf-8") as f:
        json.dump(articles, f)

    # --- NewsCLIPpings annotations per split ------------------------------- #
    split_sizes: dict[Split, int] = {
        "train": n_articles,
        "val": max(2, n_articles // 4),
        "test": max(2, n_articles // 4),
    }
    ann_dir = data_root / "news_clippings" / "data" / subset
    ann_dir.mkdir(parents=True, exist_ok=True)
    for split, size in split_sizes.items():
        annotations = []
        for j in range(size):
            cap_id = rng.randrange(n_articles)
            falsified = j % 2 == 1
            if falsified:
                # Pick a different article's image to create the mismatch.
                img_id = rng.randrange(n_articles)
                while img_id == cap_id:
                    img_id = rng.randrange(n_articles)
            else:
                img_id = cap_id
            annotations.append(
                {
                    "id": cap_id,
                    "image_id": img_id,
                    "similarity_score": round(rng.uniform(0.0, 1.0), 4),
                    "falsified": falsified,
                }
            )
        with open(ann_dir / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump({"annotations": annotations}, f)

    print(
        f"Generated dummy dataset at {data_root} "
        f"({n_articles} articles, subset='{subset}')."
    )
    return data_root


# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #


def sanity_check(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    subset: str = DEFAULT_SUBSET,
    split: Split = "train",
    n: int = 5,
) -> None:
    """Print the first ``n`` samples so you can eyeball the data wiring."""
    try:
        ds = NewsCLIPpingsDataset(data_root, split=split, subset=subset)
    except FileNotFoundError as e:
        print(f"[sanity_check] Dataset not available: {e}")
        download_newsclippings(data_root)
        return

    label_name = {LABEL_PRISTINE: "pristine", LABEL_FALSIFIED: "falsified"}
    print(f"[sanity_check] {subset}/{split}: {len(ds)} samples. First {n}:\n")
    for i in range(min(n, len(ds))):
        image_path, caption, label = ds[i]
        exists = "ok" if Path(image_path).exists() else "MISSING"
        print(f"  [{i}] label={label} ({label_name[label]})")
        print(f"      caption: {caption}")
        print(f"      image  : {image_path}  [{exists}]\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NewsCLIPpings dataset utilities.")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Generate a tiny synthetic dataset and run the sanity check on it.",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Dataset root (default: <project>/data).",
    )
    args = parser.parse_args()

    if args.dummy:
        dummy_root = Path(args.data_root) / "_dummy"
        generate_dummy_dataset(dummy_root)
        sanity_check(dummy_root)
    else:
        sanity_check(args.data_root)
