import os

# Path to the whisper-cli executable
WHISPER_CLI_PATH = r"C:\whisper\whisper-cli.exe"

# Local directory for intermediate files
TEMP_DIR = r"C:\whisper\temp"

# Available Whisper models
MODEL_MAP = {
    "Hindi-Hinglish": "ggml-hindi2hinglish-apex-q5_1.bin",
    "Medium": "ggml-medium-q5_0.bin",
}

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma4:31b-cloud"

# Prompt for the Hinglish editor
HINGLISH_PROMPT_TEMPLATE = (
    "You are an expert Hinglish editor. Clean up this raw audio transcript. "
    "Fix typos, slang words, and grammar. Keep any english words if used "
    "don't try to transalate into hinglish and keep punctuations if any. "
    "Return the output strictly as plain paragraph text only. Do NOT use "
    "markdown headers, bold stars (**), or bullet points (*). Do not add "
    "intro/outro comments, return only the raw modified text text:\n\n{text}"
)

# Prompt for the Devnagari converter
DEVNAGARI_PROMPT_TEMPLATE = (
    "You are an expert linguist. Convert the provided Romanized Hinglish text into Hindi script. "
    "Rules:\n"
    "{conversion_rule}\n"
    "3. Maintain the original meaning, grammar, and punctuation.\n"
    "4. Return ONLY the converted text. No intro, no outro, no explanations.\n\n"
    "Text:\n{text}"
)
