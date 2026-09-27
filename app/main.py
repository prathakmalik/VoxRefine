from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
import tkinter as tk
from tkinter import filedialog
from contextlib import asynccontextmanager
from app.config import MODEL_MAP, TEMP_DIR
from app.pipeline import TranscriptionPipeline, VoxRefineError, DependencyError, ModelNotFoundError, ExternalServiceError, ProcessingError
from app.utils.health import get_system_health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup: Perform system health check
    health = get_system_health()
    if health["status"] == "unhealthy":
        print("--- SYSTEM HEALTH WARNING ---")
        for missing in health["missing"]:
            print(f"Missing dependency: {missing}")
        print("Please run 'uv run python scripts/setup.py' to fix these issues.")
        print("----------------------------")
    yield
    # Shutdown logic can go here if needed

app = FastAPI(lifespan=lifespan)

# Initialize pipeline

# Serve static files for the frontend
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class TranscribeRequest(BaseModel):
    path: str
    model: str

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
    root.withdraw() # Hide the main tkinter window
    root.attributes("-topmost", True) # Bring dialog to front

    file_path = filedialog.askopenfilename(title="Select Source Video/Audio File")
    root.destroy()

    if not file_path:
        raise HTTPException(status_code=400, detail="No file selected")

    return {"path": file_path}

@app.post("/transcribe")
async def transcribe(request: TranscribeRequest):
    """Run the transcription and first-stage refinement (Hinglish)."""
    if not os.path.exists(request.path):
        raise HTTPException(status_code=400, detail=f"Source file not found at path: {request.path}")

    try:
        refined_text = pipeline.run_pipeline(request.path, request.model, MODEL_MAP)
        return {"text": refined_text}
    except DependencyError as e:
        raise HTTPException(status_code=503, detail=f"System dependency missing: {str(e)}")
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    except ExternalServiceError as e:
        raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {str(e)}")
    except ProcessingError as e:
        raise HTTPException(status_code=422, detail=f"Processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/convert-devnagari")
async def convert_devnagari(request: dict):
    """Convert cleaned Hinglish text to Devnagari."""
    text = request.get("text")
    full_conversion = request.get("full_conversion", False)

    if not text:
        raise HTTPException(status_code=400, detail="No text provided for conversion")

    try:
        devnagari_text = pipeline.convert_to_devnagari(text, full_conversion)
        return {"text": devnagari_text}
    except ExternalServiceError as e:
        raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {str(e)}")
    except ProcessingError as e:
        raise HTTPException(status_code=422, detail=f"Processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
