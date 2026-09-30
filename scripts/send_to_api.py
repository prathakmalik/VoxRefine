import json
import logging
import sys
import urllib.error
import urllib.request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SendToAPI")

# Check for debug flag in arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args, unknown = parser.parse_known_args()

if args.debug:
    logger.setLevel(logging.DEBUG)
    logger.debug("DEBUG MODE ENABLED")

try:
    logger.info("Reading input file...")
    with open("samples/input_ready.wav.txt", "r", encoding="utf-8") as f:
        input_text = f.read()
    logger.debug(f"Input text loaded. Length: {len(input_text)} chars")

    prompt = f"You are an expert Hinglish editor. Clean up this raw audio transcript. Fix typos, slang words, and grammar. Keep any english words if used don't try to transalate into hinglish and keep punctuations if any. Return the output strictly as plain paragraph text only. Do NOT use markdown headers, bold stars (**), or bullet points (*). Do not add intro/outro comments, return only the raw modified text text:\n\n{input_text}"

    body = {
        "model": "gemma4:31b-cloud",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,  # Low temperature stops creative hallucinations
            "top_p": 0.1,  # Forces the model to choose only the most accurate words
        },
    }

    logger.info("Preparing request to Ollama API...")
    logger.debug(f"Request body: {json.dumps(body, indent=2)}")

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    logger.info("Sending request to Ollama...")
    with urllib.request.urlopen(req, timeout=30) as response:
        logger.debug(f"Response status: {response.status}")
        res_data = json.loads(response.read().decode("utf-8"))
        cleaned_text = res_data.get("response", "")

        if not cleaned_text:
            logger.warning("Ollama returned an empty response.")

        logger.info("Saving cleaned transcript to file...")
        with open(
            "samples/final_cleaned_transcript.txt", "w", encoding="utf-8"
        ) as out_f:
            out_f.write(cleaned_text)

    logger.info("Done! Cleaned transcript saved successfully.")

except FileNotFoundError:
    logger.error("Input file 'samples/input_ready.wav.txt' not found.")
except urllib.error.URLError as e:
    logger.error(f"Network error connecting to Ollama: {e.reason}")
except json.JSONDecodeError:
    logger.error("Failed to decode JSON response from Ollama.")
except Exception as e:
    logger.exception(f"Unexpected error occurred: {str(e)}")
