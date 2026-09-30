# VoxRefine 🎙️

VoxRefine is a professional audio-to-text pipeline designed to handle the complexities of code-switching (e.g., Hinglish) and script conversion. It takes raw audio, transcribes it using Whisper, cleans the grammar and slang using AI, and can convert the final output into any native script (like Devnagari, Tamil, Telugu, etc.).

## 🚀 Features
- **Hardware-Aware Setup**: Automatically detects NVIDIA GPUs and installs CUDA-accelerated binaries for maximum speed.
- **Intelligent Refinement**: Uses LLMs (via Ollama) to fix typos, slang, and grammar in Romanized scripts.
- **Dynamic Multi-Script Support**: Convert Romanized Hinglish into any native script configured in the settings, with options to either keep English technical terms or perform a full conversion.
- **Process Control**: Built-in "Stop Processing" functionality to immediately terminate heavy local subprocesses (Whisper/FFmpeg) and free up GPU resources.
- **Modular Design**: Configurable model paths and prompts, making it adaptable for various languages.
- **Local & Private**: Everything runs locally on your machine.

## 🛠️ Prerequisites
- **Python 3.10+**
- **uv (Optional)**: Fast Python package manager (`pip install uv`). Recommended for faster dependency synchronization.
- **Ollama**: For AI refinement. Install from [ollama.com](https://ollama.com).
  - **Local Access**: Works by default using your local Ollama instance.
  - **Cloud Access**: Can be configured via the UI using an Ollama API Key to access cloud-hosted models.
  This project is configured to use a cloud-hosted model (`gemma4:31b-cloud`), so no local pull is required.
- **FFmpeg**: Must be installed and available in your system PATH. 
  - For Windows, download a static build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or [BtbN](https://github.com/BtbN/FFmpeg-Builds/releases), extract it, and add the `bin` folder to your system Environment Variables.

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prathakmalik/VoxRefine.git
   cd VoxRefine
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   # OR using standard pip:
   pip install -r requirements.txt
   ```

3. **Initialize the engine (Binaries & Models)**:
   Run the setup script. It will detect your GPU and ask which models you'd like to download.
   ```bash
   uv run python scripts/setup.py
   # OR:
   python scripts/setup.py
   ```
   *Note: If the automatic binary download fails, you can manually download the `whisper-bin-x64.zip` (CPU) or `whisper-cublas-bin-x64.zip` (CUDA) from the [Whisper.cpp Releases](https://github.com/ggml-org/whisper.cpp/releases) and extract them into the `engine/` folder.*

## 🖥️ Usage

1. **Start the server**:
   ```bash
   uv run python -m app.main
   # OR:
   python -m app.main
   ```
   *(Optional) Run in debug mode for detailed system logs:*
   ```bash
   uv run python -m app.main --debug
   # OR:
   python -m app.main --debug
   ```
2. **Open the UI**:
   Navigate to `http://localhost:8000/static/index.html` in your browser.
3. **Process Audio**:
   - Select your source file using the **Browse** button.
   - Choose a Whisper model.
   - Click **Start Processing**.
   - Once the cleaned text appears, use the **Settings** tab to choose your target native script, then use the conversion options to generate the final output.

## ⚙️ Configuration
You can customize the behavior in `app/config.py` or via the **Settings** tab in the UI:
- **Ollama Settings**: Configure the API URL and API Key. Use `http://localhost:11434/api/generate` for local access or the cloud API for hosted models.
- **`MODEL_MAP`**: Add new models and their filenames here.
- **`OLLAMA_MODEL`**: Change the LLM used for refinement.
- **`TARGET_SCRIPT`**: Set the default script for conversion (e.g., "Hindi (Devnagari)", "Tamil", "Telugu").
- **`PROMPTS`**: Adjust the grammar or translation rules.

## 📂 Project Structure
- `app/`: Core application logic and FastAPI endpoints.
- `engine/`: Local binaries and `.bin` model files (managed by `setup.py`).
- `scripts/`: Automation scripts for installation and developer tools.
- `temp/`: Local cache for intermediate processing files.

## 🛠️ Developer Tools
- **`scripts/send_to_api.py`**: A standalone utility to test the Ollama API connection and refine prompt engineering without running the full pipeline. Run it using:
  ```bash
  uv run python scripts/send_to_api.py --debug
  # OR:
  python scripts/send_to_api.py --debug
  ```
