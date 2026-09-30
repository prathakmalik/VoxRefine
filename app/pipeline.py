import asyncio
import json
import logging
import os
import shutil
import urllib.error
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

    async def preprocess_audio(
        self, source_path, task_id=None, active_tasks=None, debug=False
    ):
        """
        Use ffmpeg to prepare audio: highpass/lowpass filters, 16kHz, mono, PCM 16-bit.
        """
        if debug:
            logger.debug(f"Starting preprocess_audio for task {task_id}")

        filename = f"processed_{task_id}.wav" if task_id else "processed.wav"
        target_wav = os.path.join(TEMP_DIR, filename)

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
            # Create async subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            if task_id and active_tasks is not None:
                if isinstance(active_tasks[task_id], dict):
                    active_tasks[task_id]["process"] = process
                else:
                    active_tasks[task_id] = process

            # Wait for process to complete with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=3600
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise ProcessingError("FFmpeg processing timed out after 1 hour")

            if process.returncode != 0:
                raise ProcessingError(
                    f"FFmpeg processing failed: {stderr.decode() if stderr else 'Unknown error'}"
                )

        except FileNotFoundError:
            raise DependencyError(
                "FFmpeg not found. Please install FFmpeg and add it to your system PATH."
            )
        except Exception as e:
            if isinstance(e, ProcessingError):
                raise e
            raise ProcessingError(f"Unexpected FFmpeg error: {str(e)}")

        return target_wav

    async def transcribe(
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

        cmd = [WHISPER_CLI_PATH, "-m", model_path, "-f", wav_path, "-otxt"]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            if task_id and active_tasks is not None:
                if isinstance(active_tasks[task_id], dict):
                    active_tasks[task_id]["process"] = process
                else:
                    active_tasks[task_id] = process

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=3600
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise ProcessingError("Whisper transcription timed out after 1 hour")

            if process.returncode != 0:
                raise ProcessingError(
                    f"Whisper transcription failed: {stderr.decode() if stderr else 'Unknown error'}"
                )
        except Exception as e:
            if isinstance(e, ProcessingError):
                raise e
            raise ProcessingError(f"Unexpected error during transcription: {str(e)}")

        txt_path = wav_path + ".txt"
        if not os.path.exists(txt_path):
            raise ProcessingError(f"Transcription output file not found: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        return raw_text

    async def refine_text(self, raw_text):
        """
        Always first clean the text to fix typos/slang/grammar.
        """
        # Since urllib is synchronous, we wrap the call in a thread to keep the loop free
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._call_ollama, HINGLISH_PROMPT_TEMPLATE.format(text=raw_text)
        )

    async def convert_to_native_script(self, cleaned_text, full_conversion=False):
        """
        Translate and convert cleaned Romanized Hinglish text to the configured native script.
        """
        script_name = TARGET_SCRIPT if TARGET_SCRIPT else "Hindi (Devnagari)"

        # Professional role based on the target language
        system_role = f"You are a professional translator specializing in translating Romanized Hinglish into {script_name}. Your goal is to provide a perfect translation that maintains the original meaning and tone."

        # Determine the rules based on the conversion type
        if full_conversion:
            rule = (
                f"1. ABSOLUTE REQUIREMENT: Translate the entire text into formal {script_name}. "
                f"2. Use the native script of {script_name}. "
                f"3. Ensure no Hinglish or Romanized Hindi terms remain. "
                f"4. If the native script is NOT Latin, do NOT use any Latin characters."
            )
        else:
            rule = (
                f"1. ABSOLUTE REQUIREMENT: Translate the meaning of the Romanized Hinglish text into the {script_name} language and its native script. "
                f"2. If {script_name} uses a non-Latin script, do NOT use any Latin characters (except for technical terms or brands). "
                f"3. Maintain the original meaning, tone, and punctuation."
            )

        # Provide a concrete example of the translation process to anchor the model
        # We use a simple example and tell the model to apply the SAME LOGIC to the target language
        examples = (
            "Translation Example (Hinglish to English):\n"
            "Input: 'Mere ghar mein laptop hai'\n"
            "Output: 'I have a laptop in my house.'\n\n"
            f"Now, apply this same translation logic to output the text in {script_name}."
        )

        # Reinforced prompt to fight LLM bias and prevent language drift
        prompt = (
            f"{system_role}\n\n"
            f"TASK: Translate the following Romanized Hinglish text into {script_name}.\n\n"
            f"CRITICAL CONSTRAINTS:\n{rule}\n"
            f"5. OUTPUT ONLY the translated text in {script_name}. Do not include any explanations, introductory text, or comments.\n"
            f"6. If you output any text in a language other than {script_name}, the task is failed.\n\n"
            f"REFERENCE EXAMPLE:\n{examples}\n\n"
            f"Text to translate:\n{cleaned_text}"
        )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call_ollama, prompt)

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

    async def run_pipeline(
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
        import time

        start_time = time.time()

        if debug:
            logger.debug(f"Running pipeline for task {task_id}")
        if model_key not in model_map:
            raise ValueError(f"Invalid model key: {model_key}")

        model_info = model_map[model_key]
        model_filename = (
            model_info["file"] if isinstance(model_info, dict) else model_info
        )

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

        if active_tasks is not None:
            active_tasks[task_id] = {"status": "ffmpeg", "start_time": start_time}

        wav_path = await self.preprocess_audio(
            source_path, task_id, active_tasks, debug=debug
        )

        if active_tasks is not None:
            active_tasks[task_id]["status"] = "whisper"

        raw_text = await self.transcribe(
            wav_path, model_path, task_id, active_tasks, debug=debug
        )

        if active_tasks is not None:
            active_tasks[task_id]["status"] = "ollama"

        refined_text = await self.refine_text(raw_text)

        total_duration = time.time() - start_time
        return refined_text, total_duration
        total_duration = time.time() - start_time
        return refined_text, total_duration
