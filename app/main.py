from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
import tkinter as tk
from tkinter import filedialog
from app.config import MODEL_MAP, TEMP_DIR
from app.pipeline import TranscriptionPipeline

app = FastAPI()

# Initialize pipeline
pipeline = TranscriptionPipeline()

# Serve static files for the frontend
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class TranscribeRequest(BaseModel):
    path: str
    model: str

@app.get("/models")
async def get_models():
    """Return the list of available whisper models."""
    return {"models": list(MODEL_MAP.keys())}

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
