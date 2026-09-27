import os
import shutil
import subprocess
import urllib.request
import zipfile
import json
from pathlib import Path
from app.config import PROJECT_ROOT, MODELS_DIR

# Configuration for binaries
WHISPER_RELEASE_BASE = "https://github.com/ggml-org/whisper.cpp/releases/latest/download"
BINARIES_MAP = {
    "cuda": "whisper-cublas-bin-x64.zip",
    "cpu": "whisper-bin-x64.zip"
}

# Model Download URLs (Example mappings - users can edit these)
# In a real scenario, these would be direct download links to .bin files
MODEL_DOWNLOADS = {
    "ggml-medium-q5_0.bin": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin",
    "ggml-hindi2hinglish-apex-q5_1.bin": "https://huggingface.co/your-model-repo/ggml-hindi2hinglish-apex-q5_1.bin",
}

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

    if not (engine_dir / "whisper-cli.exe").exists():
        download_file(bin_url, bin_zip)
        with zipfile.ZipFile(bin_zip, 'r') as zip_ref:
            zip_ref.extractall(engine_dir)
        os.remove(bin_zip)
        print("Binaries installed successfully.")
    else:
        print("Binaries already present. Skipping download.")

    # 2. Handle Models
    models_dir = PROJECT_ROOT / "engine" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Model Selection ---")
    available_models = list(MODEL_DOWNLOADS.keys())
    for i, m in enumerate(available_models):
        print(f"[{i}] {m}")

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

    for model_file in to_download:
        dest_path = models_dir / model_file
        if not dest_path.exists():
            download_file(MODEL_DOWNLOADS[model_file], dest_path)
        else:
            print(f"{model_file} already exists. Skipping.")

    print("\nSetup complete! You can now run: uv run python -m app.main")

if __name__ == "__main__":
    setup_engine()
