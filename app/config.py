import json
from pathlib import Path

# Project root is the directory containing the 'app' folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the whisper-cli executable (Relative to root)
WHISPER_CLI_PATH = str(PROJECT_ROOT / "engine" / "whisper-cli.exe")

# Binary Download Configuration
# Update this URL when a new whisper.cpp release is available
WHISPER_RELEASE_BASE = "https://github.com/ggml-org/whisper.cpp/releases/download/b5130"

# Local directory for intermediate files
TEMP_DIR = str(PROJECT_ROOT / "temp")

# Model directory
MODELS_DIR = PROJECT_ROOT / "engine" / "models"

# Available Whisper models
# Users can add more to this map.
# Key: Display name in UI
# Value: Dict containing 'file' (filename) and optional 'url' (download link)
MODEL_MAP = {
    "Hindi-Hinglish-q5": {
        "file": "ggml-hindi2hinglish-apex-q5_1.bin",
        "url": "https://huggingface.co/voquill/whisper-hindi2hinglish-apex-ggml/resolve/main/ggml-hindi2hinglish-apex-q5_1.bin?download=true",
    },
    "Medium-q5": {
        "file": "ggml-medium-q5_0.bin",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin",
    },
    "Marquestra-Hinglish-q5": {
        "file": "ggml-apex-hinglish-q5_0.bin",
        "url": "https://huggingface.co/Marquestra/Whisper-Hindi2Hinglish-Apex-GGML/resolve/main/ggml-apex-hinglish-q5_0.bin",
    },
}

# Default Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma4:31b-cloud"
DEFAULT_TARGET_SCRIPT = "Hindi (Devnagari)"

# Load overrides from settings.json if it exists
SETTINGS_FILE = PROJECT_ROOT / "settings.json"
settings = {}
if SETTINGS_FILE.exists():
    try:
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
            OLLAMA_API_URL = settings.get("OLLAMA_API_URL", OLLAMA_API_URL)
            OLLAMA_MODEL = settings.get("OLLAMA_MODEL", OLLAMA_MODEL)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Failed to load settings.json: {e}")

TARGET_SCRIPT = settings.get("TARGET_SCRIPT", DEFAULT_TARGET_SCRIPT)

# Refinement Prompts
HINGLISH_PROMPT_TEMPLATE = (
    "You are an expert Hinglish editor. Clean up this raw audio transcript. "
    "Fix typos, slang words, and grammar. Keep any english words if used "
    "don't try to transalate into hinglish and keep punctuations if any. "
    "Return the output strictly as plain paragraph text only. Do NOT use "
    "markdown headers, bold stars (**), or bullet points (*). Do not add "
    "intro/outro comments, return only the raw modified text text:\n\n{text}"
)
