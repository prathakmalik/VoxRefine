import json
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is the directory containing the 'app' folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the whisper-cli executable (Relative to root)
WHISPER_CLI_PATH = str(PROJECT_ROOT / "engine" / "whisper-cli.exe")

# Binary Download Configuration
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
    BINARIES_MAP = {
        "cuda": {"file": "whisper-cublas-12.4.0-bin-x64.zip", "sha256": ""},
        "cpu": {"file": "whisper-bin-x64.zip", "sha256": ""},
    }
    MODEL_MAP = {}


class Settings(BaseSettings):
    """Application settings managed via environment variables and settings.json."""
    # These will be loaded from .env or environment variables first
    OLLAMA_API_URL: str = "http://localhost:11434/api/generate"
    OLLAMA_MODEL: str = "gemma4:31b-cloud"
    OLLAMA_API_KEY: Optional[str] = None
    TARGET_SCRIPT: str = "Hindi (Devnagari)"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def save(self):
        """Save current settings to settings.json for UI persistence."""
        settings_path = PROJECT_ROOT / "settings.json"
        with open(settings_path, "w") as f:
            # We only save preference-like settings to JSON, 
            # secrets should stay in .env
            data = self.model_dump()
            # Optional: you could remove OLLAMA_API_KEY from here if you want 
            # it strictly in .env, but keeping it allows the UI to show/edit it.
            json.dump(data, f, indent=4)

    @classmethod
    def load_from_json(cls):
        """
        Load settings. 
        Priority: Environment Variables (.env) > settings.json > Defaults.
        """
        # 1. Start with Pydantic's default loading (Env vars and .env file)
        settings = cls()
        
        # 2. Override with settings.json if it exists (UI preferences take priority)
        settings_path = PROJECT_ROOT / "settings.json"
        if settings_path.exists():
            try:
                with open(settings_path, "r") as f:
                    data = json.load(f)
                    # Update the settings object with values from JSON
                    for key, value in data.items():
                        if hasattr(settings, key):
                            setattr(settings, key, value)
            except Exception as e:
                print(f"Warning: Failed to load settings.json: {e}")
        
        return settings


# Global settings instance
settings = Settings.load_from_json()

# Refinement Prompts
HINGLISH_PROMPT_TEMPLATE = (
    "You are an expert Hinglish editor. Clean up this raw audio transcript. "
    "Fix typos, slang words, and grammar. Keep any english words if used "
    "don't try to transalate into hinglish and keep punctuations if any. "
    "Return the output strictly as plain paragraph text only. Do NOT use "
    "markdown headers, bold stars (**), or bullet points (*). Do not add "
    "intro/outro comments, return only the raw modified text text:\n\n{text}"
)
