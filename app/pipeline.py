import json
import logging
import os
import subprocess
import time
import urllib.request

from app.config import (
    HINGLISH_PROMPT_TEMPLATE,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    TARGET_SCRIPT,
    TEMP_DIR,
    WHISPER_CLI_PATH,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VoxRefine")


class VoxRefineError(Exception):
    """Base exception for VoxRefine pipeline errors."""

    pass


class DependencyError(VoxRefineError):
    """Raised when a system dependency (ffmpeg, binaries) is missing."""

    pass


class ModelNotFoundError(VoxRefineError):
    """Raised when a whisper model file is not found."""

    pass


class ExternalServiceError(VoxRefineError):
    """Raised when Ollama API is unavailable or returns an error."""

    pass


class ProcessingError(VoxRefineError):
    """Raised when transcription or refinement fails."""

    pass


class TranscriptionPipeline:
    def __init__(self):
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)

    def preprocess_audio(
        self, source_path, task_id=None, active_tasks=None, debug=False
    ):
        """
        Use ffmpeg to prepare audio: highpass/lowpass filters, 16kHz, mono, PCM 16-bit.
        """
        if debug:
            logger.debug(f"Starting preprocess_audio for task {task_id}")
        
        # Use task_id to create a unique filename and prevent collisions
        filename = f"processed_{task_id}.wav" if task_id else "processed.wav"
        target_wav = os.path.join(TEMP_DIR, filename)
        
        # Overwrite existing file
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            source_path,
            "-af",
            "highpass=f=200, lowpass=f=3000",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            target_wav,
        ]

        try:
            # Use Popen instead of run to allow termination
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if task_id and active_tasks is not None:
                active_tasks[task_id] = process

            # Non-blocking check for termination while waiting
            while process.poll() is None:
                # Small sleep to prevent CPU pinning
                time.sleep(0.1)

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise ProcessingError(f"FFmpeg processing failed: {stderr}")

        except FileNotFoundError:
            raise DependencyError(
                "FFmpeg not found. Please install FFmpeg and add it to your system PATH."
            )
        except Exception as e:
            raise ProcessingError(f"Unexpected FFmpeg error: {str(e)}")

        return target_wav

    def transcribe(
        self, wav_path, model_path, task_id=None, active_tasks=None, debug=False
    ):
        """
        Use whisper-cli.exe to transcribe wav to text.
        """
        if debug:
            logger.debug(f"Starting transcribe for task {task_id}")
        if not os.path.exists(WHISPER_CLI_PATH):
            raise DependencyError(
                f"Whisper binary not found at {WHISPER_CLI_PATH}. Please run setup.py."
            )

        # whisper-cli.exe -m [model] -f [wav] -otxt
        cmd = [WHISPER_CLI_PATH, "-m", model_path, "-f", wav_path, "-otxt"]

        try:
            # Use Popen instead of run to allow termination
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if task_id and active_tasks is not None:
                active_tasks[task_id] = process

            # Non-blocking check for termination while waiting
            while process.poll() is None:
                import time

                time.sleep(0.1)

            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise ProcessingError(f"Whisper transcription failed: {stderr}")
        except Exception as e:
            raise ProcessingError(f"Unexpected error during transcription: {str(e)}")

        # Whisper-cli typically creates a file with the same base name as the input + .txt
        # If input is processed.wav, output is usually processed.wav.txt
        txt_path = wav_path + ".txt"
        if not os.path.exists(txt_path):
            raise ProcessingError(f"Transcription output file not found: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        return raw_text

    def refine_text(self, raw_text):
        """
        Always first clean the text to fix typos/slang/grammar.
        """
        return self._call_ollama(HINGLISH_PROMPT_TEMPLATE.format(text=raw_text))

    def convert_to_devnagari(self, cleaned_text, full_conversion=False):
        """
        Convert cleaned Romanized Hinglish to the configured native script.
        """
        # Use the dynamic TARGET_SCRIPT from config
        script_name = TARGET_SCRIPT if TARGET_SCRIPT else "Hindi (Devnagari)"

        rule = (
            f"1. ABSOLUTE REQUIREMENT: Convert ALL words (including English) into the {script_name} script. Do NOT use Latin or Roman characters under any circumstances."
            if full_conversion
            else f"1. ABSOLUTE REQUIREMENT: Convert Hindi/Hinglish words into the {script_name} script. 2. Keep technical, brand, or proper English words in English (Latin script)."
        )

        # Few-shot examples to anchor the model to the correct script
        examples = (
            f"Example 1 (Partial):\nInput: 'Mere ghar mein laptop hai'\nOutput: 'मेरे घर में laptop है' (If {script_name} is Hindi)\n\n"
            f"Example 2 (Full):\nInput: 'Mere ghar mein laptop hai'\nOutput: 'मेरे घर में लैपटॉप है' (If {script_name} is Hindi)\n\n"
            "Note: Always match the exact script requested in the TARGET_SCRIPT field."
        )

        # Reinforced prompt to fight LLM bias
        prompt = (
            f"You are a highly skilled linguist specializing in the {script_name} script. "
            f"Your task is to translate the following Romanized Hinglish text into the {script_name} script.\n\n"
            f"CRITICAL CONSTRAINTS:\n{rule}\n"
            "3. Maintain the original meaning and punctuation.\n"
            f"4. OUTPUT ONLY the converted text in {script_name} script. Do not include any English explanations, introductory text, or comments.\n"
            f"5. If you output any text in a script other than {script_name}, the task is failed.\n\n"
            f"REFERENCE EXAMPLES:\n{examples}\n\n"
            f"Text to convert:\n{cleaned_text}"
        )
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt):
        """
        Internal helper to handle the actual API request to Ollama.
        """
        body = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}

        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("response", "")
        except urllib.error.URLError as e:
            raise ExternalServiceError(
                f"Unable to connect to Ollama server: {e.reason}"
            )
        except Exception as e:
            raise ExternalServiceError(f"Ollama API error: {str(e)}")

    def cleanup_task(self, task_id):
        """
        Remove all temporary files associated with a specific task_id.
        """
        if not task_id:
            return

        try:
            for filename in os.listdir(TEMP_DIR):
                if filename.startswith(f"processed_{task_id}"):
                    file_path = os.path.join(TEMP_DIR, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            logger.debug(f"Cleaned up temporary files for task {task_id}")
        except Exception as e:
            logger.error(f"Error cleaning up task {task_id}: {e}")

    def run_pipeline(
        self,
        source_path,
        model_key,
        model_map,
        task_id=None,
        active_tasks=None,
        debug=False,
    ):
        """
        Coordinate the first part: Preprocess -> Transcribe -> Refine (Hinglish).
        """
        if debug:
            logger.debug(f"Running pipeline for task {task_id}")
        if model_key not in model_map:
            raise ValueError(f"Invalid model key: {model_key}")

        # Handle the new MODEL_MAP structure where value is a dict with 'file' and 'url'
        model_info = model_map[model_key]
        model_filename = (
            model_info["file"] if isinstance(model_info, dict) else model_info
        )

        # Ensure we have the absolute path to the model file
        from app.config import MODELS_DIR

        model_path = (
            str(MODELS_DIR / model_filename)
            if not os.path.isabs(model_filename)
            else model_filename
        )

        if not os.path.exists(model_path):
            raise ModelNotFoundError(
                f"Model file {model_filename} not found at {model_path}. "
                f"Please run 'uv run python scripts/setup.py' to download the required models."
            )

        # 1. Preprocess
        wav_path = self.preprocess_audio(
            source_path, task_id, active_tasks, debug=debug
        )

        # 2. Transcribe
        raw_text = self.transcribe(
            wav_path, model_path, task_id, active_tasks, debug=debug
        )

        # 3. Refine (Always start with Hinglish clean)
        refined_text = self.refine_text(raw_text)

        return refined_text
