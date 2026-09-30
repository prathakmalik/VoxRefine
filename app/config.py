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

# Load models and binaries from external JSON configuration
MODELS_CONFIG_PATH = PROJECT_ROOT / "models.json"
if MODELS_CONFIG_PATH.exists():
    with open(MODELS_CONFIG_PATH, "r") as f:
        config_data = json.load(f)
        BINARIES_MAP = config_data.get("binaries", {})
        MODEL_MAP = config_data.get("models", {})
else:
    # Fallback defaults if models.json is missing
    BINARIES_MAP = {
        "cuda": {"file": "whisper-cublas-12.4.0-bin-x64.zip", "sha256": ""},
        "cpu": {"file": "whisper-bin-x64.zip", "sha256": ""},
    }
    MODEL_MAP = {}

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
