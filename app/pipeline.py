import subprocess
import os
import json
import urllib.request
from app.config import WHISPER_CLI_PATH, TEMP_DIR, OLLAMA_API_URL, OLLAMA_MODEL, HINGLISH_PROMPT_TEMPLATE, DEVNAGARI_PROMPT_TEMPLATE, MODELS_DIR

class TranscriptionPipeline:
    def __init__(self):
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)

    def preprocess_audio(self, source_path):
        """
        Use ffmpeg to prepare audio: highpass/lowpass filters, 16kHz, mono, PCM 16-bit.
        """
        target_wav = os.path.join(TEMP_DIR, "processed.wav")
        # Overwrite existing file
        cmd = [
            "ffmpeg", "-y",
            "-i", source_path,
            "-af", "highpass=f=200, lowpass=f=3000",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            target_wav
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")

        return target_wav

    def transcribe(self, wav_path, model_path):
        """
        Use whisper-cli.exe to transcribe wav to text.
        """
        # whisper-cli.exe -m [model] -f [wav] -otxt
        cmd = [
            WHISPER_CLI_PATH,
            "-m", model_path,
            "-f", wav_path,
            "-otxt"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Whisper error: {result.stderr}")

        # Whisper-cli typically creates a file with the same base name as the input + .txt
        # If input is processed.wav, output is usually processed.wav.txt
        txt_path = wav_path + ".txt"
        if not os.path.exists(txt_path):
            raise Exception(f"Transcription file not found: {txt_path}")

        with open(txt_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        return raw_text

    def refine_text(self, raw_text):
        """
        Always first clean the text to fix typos/slang/grammar.
        """
        return self._call_ollama(HINGLISH_PROMPT_TEMPLATE.format(text=raw_text))

    def convert_to_devnagari(self, cleaned_text, full_conversion=False):
        """
        Convert cleaned Romanized Hinglish to Devnagari.
        """
        rule = (
            "1. Convert ALL words (including English) into Devnagari script."
            if full_conversion else
            "1. Convert Hindi/Hinglish words into Devnagari script.\n2. Keep technical, brand, or proper English words in English (Latin script)."
        )

        prompt = DEVNAGARI_PROMPT_TEMPLATE.format(conversion_rule=rule, text=cleaned_text)
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt):
        """
        Internal helper to handle the actual API request to Ollama.
        """
        body = {
            'model': OLLAMA_MODEL,
            'prompt': prompt,
            'stream': False
        }

        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=json.dumps(body).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data.get('response', '')
        except Exception as e:
            raise Exception(f"Ollama API error: {e}")

    def run_pipeline(self, source_path, model_key, model_map):
        """
        Coordinate the first part: Preprocess -> Transcribe -> Refine (Hinglish).
        """
        if model_key not in model_map:
            raise ValueError(f"Invalid model key: {model_key}")

        # Handle the new MODEL_MAP structure where value is a dict with 'file' and 'url'
        model_info = model_map[model_key]
        model_filename = model_info['file'] if isinstance(model_info, dict) else model_info

        # Ensure we have the absolute path to the model file
        from app.config import MODELS_DIR
        model_path = str(MODELS_DIR / model_filename) if not os.path.isabs(model_filename) else model_filename

        # 1. Preprocess
        wav_path = self.preprocess_audio(source_path)

        # 2. Transcribe
        raw_text = self.transcribe(wav_path, model_path)

        # 3. Refine (Always start with Hinglish clean)
        refined_text = self.refine_text(raw_text)

        return refined_text
