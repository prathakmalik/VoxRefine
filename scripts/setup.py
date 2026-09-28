import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# Ensure the project root is in the python path so we can import from app.config
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app.config import MODEL_MAP, PROJECT_ROOT, WHISPER_RELEASE_BASE

# Configuration for binaries
BINARIES_MAP = {
    "cuda": "whisper-cublas-12.4.0-bin-x64.zip",
    "cpu": "whisper-bin-x64.zip",
}


def check_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError, FileNotFoundError:
        return False


def download_file(url, destination):
    print(f"Downloading {url} ...")
    try:
        with urllib.request.urlopen(url) as response:
            total_size = response.getheader("Content-Length")
            if total_size:
                total_size = int(total_size)

            with open(destination, "wb") as out_file:
                downloaded = 0
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        percent = (downloaded / total_size) * 100
                        downloaded_mb = downloaded / (1024 * 1024)
                        total_size_mb = total_size / (1024 * 1024)
                        sys.stdout.write(f"\rProgress: {percent:.2f}% ({downloaded_mb:.2f}/{total_size_mb:.2f} MB)")
                    else:
                        downloaded_mb = downloaded / (1024 * 1024)
                        sys.stdout.write(f"\rDownloaded: {downloaded_mb:.2f} MB")
                    sys.stdout.flush()
        print(f"\nSaved to {destination}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"\nError downloading {url}: {e}")
        return False


def setup_engine():
    print("--- VoxRefine Setup ---")
    success = True

    # 1. Handle Binaries
    engine_dir = PROJECT_ROOT / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)

    if (engine_dir / "whisper-cli.exe").exists() or (engine_dir / "main.exe").exists():
        print("Binaries already present. Skipping download.")
    else:
        has_gpu = check_gpu()
        print(
            f"GPU Detection: {'NVIDIA GPU Found' if has_gpu else 'No NVIDIA GPU Found'}"
        )

        binary_type = "cuda" if has_gpu else "cpu"
        if has_gpu:
            choice = input(
                "Download CUDA-accelerated binaries? (y/n, default: y): "
            ).lower()
            if choice == "n":
                binary_type = "cpu"
        else:
            print("Defaulting to CPU binaries.")

        bin_filename = BINARIES_MAP[binary_type]
        bin_url = f"{WHISPER_RELEASE_BASE}/{bin_filename}"
        bin_zip = engine_dir / bin_filename

        if download_file(bin_url, bin_zip):
            with zipfile.ZipFile(bin_zip, "r") as zip_ref:
                # Handle zip files that contain a single top-level directory
                namelist = zip_ref.namelist()
                top_level = None
                if namelist:
                    first_part = namelist[0].split('/')[0]
                    if all(name.startswith(first_part + '/') or name == first_part + '/' for name in namelist):
                        top_level = first_part

                if top_level:
                    for member in zip_ref.infolist():
                        filename = member.filename
                        if filename.startswith(top_level + '/'):
                            filename = filename[len(top_level) + 1:]
                        elif filename == top_level + '/':
                            continue
                        
                        target_path = engine_dir / filename
                        if member.is_dir():
                            target_path.mkdir(parents=True, exist_ok=True)
                        else:
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                                shutil.copyfileobj(source, target)
                else:
                    zip_ref.extractall(engine_dir)
            os.remove(bin_zip)
            print("Binaries installed successfully.")
        else:
            print("Failed to download binaries.")
            success = False

    # 2. Handle Models
    models_dir = PROJECT_ROOT / "engine" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Model Selection ---")
    available_models = list(MODEL_MAP.keys())
    for i, name in enumerate(available_models):
        print(f"[{i}] {name} ({MODEL_MAP[name]['file']})")

    selection = input(
        "Enter model indices to download (comma separated, e.g. 0,1) or 'all': "
    )

    to_download = []
    if selection.lower() == "all":
        to_download = available_models
    else:
        try:
            indices = [int(x.strip()) for x in selection.split(",")]
            to_download = [available_models[i] for i in indices]
        except ValueError, IndexError:
            print("Invalid selection. Skipping model downloads.")

    for model_name in to_download:
        model_info = MODEL_MAP[model_name]
        model_file = model_info["file"]
        url = model_info.get("url")

        dest_path = models_dir / model_file
        if not dest_path.exists():
            if url:
                if not download_file(url, dest_path):
                    success = False
            else:
                print(
                    f"No download URL provided for {model_name}. Please add it manually to engine/models/{model_file}"
                )
                success = False
        else:
            print(f"{model_file} already exists. Skipping.")

    if success:
        print("\nSetup complete! You can now run: uv run python -m app.main")
    else:
        print("\nSetup finished with errors. Please check the logs above.")


if __name__ == "__main__":
    setup_engine()
