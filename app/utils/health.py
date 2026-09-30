import shutil
import urllib.request
from pathlib import Path

from app.config import OLLAMA_API_URL, WHISPER_CLI_PATH


def check_ffmpeg():
    """Verify if ffmpeg is installed and available in PATH."""
    return shutil.which("ffmpeg") is not None


def check_whisper_binaries():
    """Verify if the whisper-cli executable exists."""
    return Path(WHISPER_CLI_PATH).exists()


def check_ollama():
    """Verify if the Ollama server is responding."""
    try:
        # Determine the base URL to check health.
        # If the URL is already a specific endpoint (like /api/generate or /api/chat),
        # we need to get to the base URL to check the root or /api/tags.
        base_url = OLLAMA_API_URL
        for endpoint in ["/api/generate", "/api/chat"]:
            if base_url.endswith(endpoint):
                base_url = base_url[: -len(endpoint)]
        
        # Try /api/tags as it's a standard lightweight endpoint
        url = base_url.rstrip('/') + "/api/tags"
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
        "missing": [k for k, v in health.items() if not v],
    }
