import json
import urllib.request

with open('samples/input_ready.wav.txt', 'r', encoding='utf-8') as f:
    input_text = f.read()

prompt = f"You are an expert Hinglish editor. Clean up this raw audio transcript. Fix typos, slang words, and grammar. Keep any english words if used don't try to transalate into hinglish and keep punctuations if any. Return the output strictly as plain paragraph text only. Do NOT use markdown headers, bold stars (**), or bullet points (*). Do not add intro/outro comments, return only the raw modified text text:\n\n{input_text}"

body = {
    'model': 'gemma4:31b-cloud',
    'prompt': prompt,
    'stream': False
}

req = urllib.request.Request(
    'http://localhost:11434/api/generate',
    data=json.dumps(body).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)

try:
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode('utf-8'))
        cleaned_text = res_data.get('response', '')
        with open('samples/final_cleaned_transcript.txt', 'w', encoding='utf-8') as out_f:
            out_f.write(cleaned_text)
    print('Done! Cleaned transcript saved successfully.')
except Exception as e:
    print(f'Error occurred: {e}')
