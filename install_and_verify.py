import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"


def run_command(args, **kwargs):
    print("Running:", " ".join(args))
    subprocess.run(args, check=True, **kwargs)


def install_requirements():
    if shutil.which("uv"):
        print("Detected uv. Installing requirements with uv...")
        run_command(["uv", "add", "-r", str(REQUIREMENTS)])
    else:
        print("uv not found. Installing requirements with pip...")
        run_command([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])


def verify_clip():
    if shutil.which("uv"):
        print("Verifying CLIP inside the uv-managed environment...")
        verify_script = (
            "import clip, torch\n"
            "from PIL import Image\n"
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "print('Using device:', device)\n"
            "model, preprocess = clip.load('ViT-B/32', device=device)\n"
            "print('CLIP model loaded successfully.')\n"
            "text_inputs = clip.tokenize(['This is a test caption.']).to(device)\n"
            "image = Image.new('RGB', (224, 224), color=(128, 128, 128))\n"
            "image_input = preprocess(image).unsqueeze(0).to(device)\n"
            "with torch.no_grad():\n"
            "    text_features = model.encode_text(text_inputs)\n"
            "    image_features = model.encode_image(image_input)\n"
            "print('Text features shape:', text_features.shape)\n"
            "print('Image features shape:', image_features.shape)\n"
            "print('CLIP verification completed successfully.')\n"
        )
        run_command(["uv", "run", "python", "-c", verify_script])
        return

    try:
        import clip
        import torch
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Missing required package after install. Please ensure the environment is active and rerun this script."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model loaded successfully.")

    text_inputs = clip.tokenize(["This is a test caption."]).to(device)
    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        image_features = model.encode_image(image_input)

    print("Text features shape:", text_features.shape)
    print("Image features shape:", image_features.shape)
    print("CLIP verification completed successfully.")


if __name__ == "__main__":
    install_requirements()
    verify_clip()
