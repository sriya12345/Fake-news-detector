from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fake-news-detector",
    version="0.1.0",
    description="Fake news detector using CLIP, Hugging Face transformers, Gradio, and scikit-learn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10,<3.15",
    install_requires=[
        "torch",
        "torchvision",
        "clip @ git+https://github.com/openai/CLIP.git",
        "transformers",
        "datasets",
        "scikit-learn",
        "numpy",
        "pandas",
        "Pillow",
        "gradio",
        "matplotlib",
        "seaborn",
        "jupyter",
        "tqdm",
    ],
    py_modules=["main"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
