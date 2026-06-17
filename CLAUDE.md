# Fake News Detector — Project Brief

## What this project is
Multimodal fake news detector using CLIP embeddings
to measure text-image semantic consistency.
Dataset: NewsCLIPpings (MIT).

## Stack
- Python 3.10+, PyTorch, CLIP (OpenAI), HuggingFace
- Gradio for deployment, scikit-learn, Pillow

## Project structure
fake-news-detector/
├── data/          # dataset download scripts
├── src/
│   ├── model.py   # CLIP embedding + classifier
│   ├── train.py   # training loop
│   └── app.py     # Gradio demo
├── notebooks/     # EDA and experiments
├── requirements.txt
└── README.md

## Current goal
Build MVP: load CLIP, extract embeddings for
image-text pairs, train MLP classifier, evaluate.