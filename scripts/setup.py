import os
import shutil
import subprocess
import urllib.request
import zipfile
import json
import sys
from pathlib import Path

# Ensure the project root is in the python path so we can import from app.config
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app.config import PROJECT_ROOT, MODELS_DIR, MODEL_MAP, WHISPER_RELEASE_BASE

# Configuration for binaries
BINARIES_MAP = {
    "cuda": "whisper-cublas-bin-x64.zip",
    "cpu": "whisper-bin-x64.zip"
}

# Model Download URLs are now managed in app/config.MODEL_MAP
# No local MODEL_DOWNLOADS dictionary needed here.

def check_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def download_file(url, destination):
    print(f"Downloading {url} ...")
    try:
        with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"Saved to {destination}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def setup_engine():
    print("--- VoxRefine Setup ---")

    # 1. Handle Binaries
    engine_dir = PROJECT_ROOT / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)

    if (engine_dir / "whisper-cli.exe").exists() or (engine_dir / "main.exe").exists():
        print("Binaries already present. Skipping download.")
    else:
        has_gpu = check_gpu()
        print(f"GPU Detection: {'NVIDIA GPU Found' if has_gpu else 'No NVIDIA GPU Found'}")

        binary_type = "cuda" if has_gpu else "cpu"
        if has_gpu:
            choice = input("Download CUDA-accelerated binaries? (y/n, default: y): ").lower()
            if choice == 'n': binary_type = "cpu"
        else:
            print("Defaulting to CPU binaries.")

        bin_filename = BINARIES_MAP[binary_type]
        bin_url = f"{WHISPER_RELEASE_BASE}/{bin_filename}"
        bin_zip = engine_dir / bin_filename

        download_file(bin_url, bin_zip)
        with zipfile.ZipFile(bin_zip, 'r') as zip_ref:
            zip_ref.extractall(engine_dir)
        os.remove(bin_zip)
        print("Binaries installed successfully.")

    # 2. Handle Models
    models_dir = PROJECT_ROOT / "engine" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Model Selection ---")
    available_models = list(MODEL_MAP.keys())
    for i, name in enumerate(available_models):
        print(f"[{i}] {name} ({MODEL_MAP[name]['file']})")

    selection = input("Enter model indices to download (comma separated, e.g. 0,1) or 'all': ")

    to_download = []
    if selection.lower() == 'all':
        to_download = available_models
    else:
        try:
            indices = [int(x.strip()) for x in selection.split(',')]
            to_download = [available_models[i] for i in indices]
        except (ValueError, IndexError):
            print("Invalid selection. Skipping model downloads.")

    for model_name in to_download:
        model_info = MODEL_MAP[model_name]
        model_file = model_info['file']
        url = model_info.get('url')

        dest_path = models_dir / model_file
        if not dest_path.exists():
            if url:
                download_file(url, dest_path)
            else:
                print(f"No download URL provided for {model_name}. Please add it manually to engine/models/{model_file}")
        else:
            print(f"{model_file} already exists. Skipping.")

    print("\nSetup complete! You can now run: uv run python -m app.main")

if __name__ == "__main__":
    setup_engine()
