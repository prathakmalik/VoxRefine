import shutil
import urllib.request
import json
from pathlib import Path
from app.config import WHISPER_CLI_PATH, OLLAMA_API_URL

def check_ffmpeg():
    """Verify if ffmpeg is installed and available in PATH."""
    return shutil.which("ffmpeg") is not None

def check_whisper_binaries():
    """Verify if the whisper-cli executable exists."""
    return Path(WHISPER_CLI_PATH).exists()

def check_ollama():
    """Verify if the Ollama server is responding."""
    try:
        # Use a simple request to the base API or a lightweight endpoint
        # Ollama usually responds to GET /api/tags or just a GET to the base URL
        url = OLLAMA_API_URL.replace("/api/generate", "/api/tags")
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False

def get_system_health():
    """
    Perform all system health checks and return a report.
    """
    health = {
        "ffmpeg": check_ffmpeg(),
        "whisper_binaries": check_whisper_binaries(),
        "ollama": check_ollama(),
    }

    status = "healthy" if all(health.values()) else "unhealthy"

    return {
        "status": status,
        "checks": health,
        "missing": [k for k, v in health.items() if not v]
    }
