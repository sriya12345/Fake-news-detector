# 📰 Multimodal Fake News Detector

Detects out-of-context misinformation by measuring the **semantic consistency
between a news image and its caption** using [CLIP](https://github.com/openai/CLIP).
A frozen CLIP backbone extracts image and text embeddings; a small MLP head
classifies the pair as *real (pristine)* or *manipulated (falsified)*.

Dataset: [NewsCLIPpings](https://github.com/g-luo/news_clippings) (built on the
[VisualNews](https://github.com/FuxiaoLiu/VisualNews-Repository) corpus).

## How it works

```
image ─┐
       ├─► CLIP (frozen) ─► [image_emb (512), text_emb (512), cosine_sim (1)] ─► MLP head ─► real / manipulated
text ──┘
```

- **Backbone:** `openai/clip-vit-base-patch32`, weights frozen.
- **Features:** L2-normalized image & text embeddings plus their cosine similarity.
- **Head:** MLP trained with binary cross-entropy (single-logit output).

## Project structure

```
fake-news-detector/
├── src/
│   ├── dataset.py   # NewsCLIPpings loader + DataLoaders (+ dummy-data generator)
│   ├── model.py     # frozen CLIP + MLP classifier
│   ├── train.py     # training loop (BCE, Adam, early stopping, W&B)
│   ├── evaluate.py  # metrics + confusion matrix + misclassified CSV
│   └── app.py       # Gradio demo
├── data/            # datasets (gitignored)
├── checkpoints/     # saved models (gitignored)
├── outputs/         # evaluation artifacts (gitignored)
├── pyproject.toml
└── requirements.txt
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/). Python 3.10–3.12 recommended.

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

> On Windows, run scripts with the venv interpreter: `./.venv/Scripts/python.exe ...`
> (or prefix commands with `uv run`).

## Quick start (no dataset needed)

Every script supports a `--dummy` flag that generates tiny synthetic data so you
can exercise the full pipeline offline:

```bash
# train 3 epochs on synthetic data, no W&B
python src/train.py --dummy --epochs 3 --wandb-mode disabled

# evaluate the resulting checkpoint
python src/evaluate.py --dummy

# launch the demo
python src/app.py
```

## Getting the real dataset

NewsCLIPpings references images from the **VisualNews** corpus, which is
access-gated. Expected layout under `data/`:

```
data/
├── visual_news/origin/data.json + images...
└── news_clippings/data/<subset>/{train,val,test}.json
```

1. NewsCLIPpings annotations: https://github.com/g-luo/news_clippings
2. VisualNews images (request access): https://github.com/FuxiaoLiu/VisualNews-Repository

Verify the wiring with `python src/dataset.py` (prints a few samples).

## Training

```bash
python src/train.py \
  --data-root data --subset merged_balanced \
  --batch-size 32 --epochs 30 --lr 1e-3 \
  --patience 5 --wandb-mode online
```

The best checkpoint (by validation loss) is saved to `checkpoints/best_model.pt`.
W&B logs `train/loss`, `train/acc`, `val/loss`, `val/acc`; use
`--wandb-mode {online,offline,disabled}`. Run `wandb login` first for online mode.

## Evaluation

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt
```

Reports accuracy, precision, recall, F1, and a confusion matrix, and writes up
to 10 misclassified examples to `outputs/misclassified.csv`.

## Demo

```bash
python src/app.py                 # http://localhost:7860
python src/app.py --share         # public link
```

Paste a headline, upload an image, and the app shows a 0–100 consistency score,
a likely-real / likely-manipulated verdict, and a heatmap of where the image
matches the caption.

## License

MIT
