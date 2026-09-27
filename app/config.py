import os
from pathlib import Path

# Project root is the directory containing the 'app' folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the whisper-cli executable (Relative to root)
WHISPER_CLI_PATH = str(PROJECT_ROOT / "engine" / "whisper-cli.exe")

# Local directory for intermediate files
TEMP_DIR = str(PROJECT_ROOT / "temp")

# Model directory
MODELS_DIR = PROJECT_ROOT / "engine" / "models"

# Available Whisper models
# Users can add more to this map. The key is displayed in UI, the value is the filename.
MODEL_MAP = {
    "Hindi-Hinglish": "ggml-hindi2hinglish-apex-q5_1.bin",
    "Medium": "ggml-medium-q5_0.bin",
}

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma4:31b-cloud"

# Refinement Prompts
HINGLISH_PROMPT_TEMPLATE = (
    "You are an expert Hinglish editor. Clean up this raw audio transcript. "
    "Fix typos, slang words, and grammar. Keep any english words if used "
    "don't try to transalate into hinglish and keep punctuations if any. "
    "Return the output strictly as plain paragraph text only. Do NOT use "
    "markdown headers, bold stars (**), or bullet points (*). Do not add "
    "intro/outro comments, return only the raw modified text text:\n\n{text}"
)

DEVNAGARI_PROMPT_TEMPLATE = (
    "You are an expert linguist. Convert the provided Romanized Hinglish text into Hindi script. "
    "Rules:\n"
    "{conversion_rule}\n"
    "3. Maintain the original meaning, grammar, and punctuation.\n"
    "4. Return ONLY the converted text. No intro, no outro, no explanations.\n\n"
    "Text:\n{text}"
)
