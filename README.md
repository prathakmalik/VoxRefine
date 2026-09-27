# VoxRefine 🎙️

VoxRefine is a professional audio-to-text pipeline designed to handle the complexities of code-switching (e.g., Hinglish) and script conversion. It takes raw audio, transcribes it using Whisper, cleans the grammar and slang using AI, and can optionally convert the final output into native scripts (like Devnagari).

## 🚀 Features
- **Hardware-Aware Setup**: Automatically detects NVIDIA GPUs and installs CUDA-accelerated binaries for maximum speed.
- **Intelligent Refinement**: Uses LLMs (via Ollama) to fix typos, slang, and grammar in Romanized scripts.
- **Multi-Script Support**: Convert Romanized Hinglish into clean Hindi (Devnagari) while optionally preserving English technical terms.
- **Modular Design**: Configurable model paths and prompts, making it adaptable for other languages (Tamil, Telugu, etc.).
- **Local & Private**: Everything runs locally on your machine.

## 🛠️ Prerequisites
- **Python 3.10+**
- **uv**: Fast Python package manager (`pip install uv`)
- **Ollama**: For AI refinement. Install from [ollama.com](https://ollama.com) and pull the required model:
  ```bash
  ollama pull gemma4:31b-cloud
  ```
- **FFmpeg**: Must be installed and available in your system PATH.

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prathakmalik/VoxRefine.git
   cd VoxRefine
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Initialize the engine (Binaries & Models)**:
   Run the setup script. It will detect your GPU and ask which models you'd like to download.
   ```bash
   uv run python scripts/setup.py
   ```

## 🖥️ Usage

1. **Start the server**:
   ```bash
   uv run python -m app.main
   ```
2. **Open the UI**:
   Navigate to `http://localhost:8000/static/index.html` in your browser.
3. **Process Audio**:
   - Select your source file using the **Browse** button.
   - Choose a Whisper model.
   - Click **Start Processing**.
   - Once the cleaned text appears, use the **Devnagari Options** to convert the script.

## ⚙️ Configuration
You can customize the behavior in `app/config.py`:
- **`MODEL_MAP`**: Add new models and their filenames here.
- **`OLLAMA_MODEL`**: Change the LLM used for refinement.
- **`PROMPTS`**: Adjust the grammar or translation rules.

## 📂 Project Structure
- `app/`: Core application logic and FastAPI endpoints.
- `engine/`: Local binaries and `.bin` model files (managed by `setup.py`).
- `scripts/`: Automation scripts for installation.
- `temp/`: Local cache for intermediate processing files.
