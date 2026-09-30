import json
import logging
import os
import shutil
import tkinter as tk
from contextlib import asynccontextmanager
from tkinter import filedialog

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import MODEL_MAP, PROJECT_ROOT, TEMP_DIR
from app.pipeline import (
    DependencyError,
    ExternalServiceError,
    ModelNotFoundError,
    ProcessingError,
    TranscriptionPipeline,
)
from app.utils.health import get_system_health

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VoxRefine")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup: Perform system health check
    health = get_system_health()
    if health["status"] == "unhealthy":
        logger.warning("--- SYSTEM HEALTH WARNING ---")
        for missing in health["missing"]:
            logger.warning(f"Missing dependency: {missing}")
        logger.warning(
            "Please run 'uv run python scripts/setup.py' to fix these issues."
        )
        logger.warning("----------------------------")
    yield
    # Shutdown logic can go here if needed


app = FastAPI(lifespan=lifespan)

import uuid

# Initialize pipeline
pipeline = TranscriptionPipeline()

# Global registry to track active subprocesses
active_tasks = {}

# Serve static files for the frontend
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class TranscribeRequest(BaseModel):
    path: str
    model: str
    task_id: str


@app.get("/models")
async def get_models():
    """Return the list of available whisper models."""
    return {"models": list(MODEL_MAP.keys())}


@app.get("/health")
async def health_check():
    """Check system dependencies and Ollama connectivity."""
    return get_system_health()


@app.get("/pick-file")
async def pick_file():
    """Trigger a native Windows file dialog to select a file path."""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes("-topmost", True)  # Bring dialog to front

    file_path = filedialog.askopenfilename(title="Select Source Video/Audio File")
    root.destroy()

    if not file_path:
        raise HTTPException(status_code=400, detail="No file selected")

    return {"path": file_path}


@app.get("/task-status")
async def get_task_status(task_id: str):
    """Return the current status of a running transcription task."""
    if task_id not in active_tasks:
        return {"status": "completed", "stage": None}
    
    task_data = active_tasks[task_id]
    if isinstance(task_data, dict):
        return {"status": "processing", "stage": task_data.get("status")}
    
    # Fallback for old registry format
    return {"status": "processing", "stage": "unknown"}

@app.post("/transcribe")
async def transcribe(request: TranscribeRequest):
    """Run the transcription and first-stage refinement (Hinglish)."""
    if not os.path.exists(request.path):
        raise HTTPException(
            status_code=400, detail=f"Source file not found at path: {request.path}"
        )

    task_id = request.task_id

    try:
        # Pass task_id and the global registry to the pipeline
        # Use a global debug flag or check the logger level
        import logging

        debug_mode = logging.getLogger("VoxRefine").getEffectiveLevel() == logging.DEBUG

        refined_text, duration = await pipeline.run_pipeline(
            request.path,
            request.model,
            MODEL_MAP,
            task_id=task_id,
            active_tasks=active_tasks,
            debug=debug_mode,
        )
        return {"text": refined_text, "task_id": task_id, "duration": duration}
    except DependencyError as e:
        raise HTTPException(
            status_code=503, detail=f"System dependency missing: {str(e)}"
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=503, detail=f"Ollama service unavailable: {str(e)}"
        )
    except ProcessingError as e:
        raise HTTPException(status_code=422, detail=f"Processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Cleanup: Remove the task from the registry and delete temporary files
        if task_id in active_tasks:
            del active_tasks[task_id]
        pipeline.cleanup_task(task_id)


@app.delete("/stop")
async def stop_task(task_id: str):
    """Terminate a running subprocess associated with a task_id."""
    import logging

    logger = logging.getLogger("VoxRefine")

    logger.info(f"Stop request received for task_id: {task_id}")

    if task_id not in active_tasks:
        logger.warning(
            f"Stop failed: Task ID {task_id} not found in active_tasks registry"
        )
        raise HTTPException(
            status_code=404, detail="Task ID not found or already completed"
        )

    try:
        # Handle both the new dict format and the old direct process format
        task_data = active_tasks[task_id]
        process = task_data["process"] if isinstance(task_data, dict) else task_data
        pid = process.pid
        logger.info(f"Attempting to terminate process {pid} for task {task_id}")

        # On Windows, killing the parent process sometimes leaves children (like whisper-cli) running.
        # We can use taskkill to kill the process tree.
        import subprocess

        kill_result = subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, text=True
        )
        logger.debug(f"Taskkill output: {kill_result.stdout} {kill_result.stderr}")

        process.kill()
        del active_tasks[task_id]
        logger.info(f"Successfully terminated process {pid} and removed from registry")
        return {"status": "Task terminated successfully"}
    except Exception as e:
        logger.exception(f"Error while stopping task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to terminate process: {str(e)}"
        )


@app.post("/convert-native-script")
async def convert_native_script(request: dict):
    """Translate and convert cleaned Hinglish text to the configured native script."""
    from app.config import TARGET_SCRIPT
    logger.info(f"Converting text to native script: {TARGET_SCRIPT}")
    
    text = request.get("text")
    full_conversion = request.get("full_conversion", False)

    if not text:
        raise HTTPException(status_code=400, detail="No text provided for conversion")

    try:
        native_text = await pipeline.convert_to_native_script(text, full_conversion)
        return {"text": native_text}
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=503, detail=f"Ollama service unavailable: {str(e)}"
        )
    except ProcessingError as e:
        raise HTTPException(status_code=422, detail=f"Processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


class SettingsRequest(BaseModel):
    OLLAMA_API_URL: str
    OLLAMA_MODEL: str
    TARGET_SCRIPT: str


@app.get("/settings")
async def get_settings():
    """Return current API and Model settings."""
    import app.config as config

    return {
        "OLLAMA_API_URL": config.OLLAMA_API_URL,
        "OLLAMA_MODEL": config.OLLAMA_MODEL,
        "TARGET_SCRIPT": config.TARGET_SCRIPT,
    }


@app.post("/settings")
async def save_settings(request: SettingsRequest):
    """Save settings to settings.json and update config."""
    try:
        settings_data = {
            "OLLAMA_API_URL": request.OLLAMA_API_URL,
            "OLLAMA_MODEL": request.OLLAMA_MODEL,
            "TARGET_SCRIPT": request.TARGET_SCRIPT,
        }
        with open(PROJECT_ROOT / "settings.json", "w") as f:
            json.dump(settings_data, f, indent=4)

        # Update the active config in memory
        import app.config as config

        config.OLLAMA_API_URL = request.OLLAMA_API_URL
        config.OLLAMA_MODEL = request.OLLAMA_MODEL
        config.TARGET_SCRIPT = request.TARGET_SCRIPT

        return {"status": "Settings saved successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save settings: {str(e)}"
        )


@app.delete("/temp")
async def clear_temp():
    """Clear all intermediate files in the temp directory."""
    try:
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        return {"status": "cleaned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear temp: {e}")


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="VoxRefine API Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        import logging

        logging.getLogger("VoxRefine").setLevel(logging.DEBUG)
        print("DEBUG MODE ENABLED")

    uvicorn.run(app, host="0.0.0.0", port=8000)
