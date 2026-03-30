# ShadowView - IQ-Edge Interview Assistant

An intelligent interview assistant that monitors your screen, extracts text via OCR, analyzes content type, and provides AI-powered responses using a local LLM (Ollama).

## Features

- 📸 **Full-screen capture** compatible with macOS, Linux, and Windows
- 🔍 **OCR text extraction** with confidence scoring (English + Spanish)
- 🧠 **Content analysis** - Detects IQ patterns, math problems, spatial/verbal questions, code
- 💡 **Dynamic prompts** - Tailored LLM instructions based on content type
- 🎯 **Local LLM integration** - Uses Ollama for privacy and speed
- 🔄 **Change detection** - Only processes when screen content changes
- 🎹 **Hotkey support** - Press F6 for manual capture (optional)

## Prerequisites

### macOS Setup

1. **Screenshot Permission**: Grant Screen Recording access to Terminal/VSCode
   - Go to: System Settings → Privacy & Security → Screen Recording
   - Add Terminal or your IDE to the allowed apps

2. **Python 3.12+**: Verify your Python version
   ```bash
   python --version  # Should be >= 3.12
   ```

3. **Tesseract OCR**: Install via Homebrew
   ```bash
   brew install tesseract
   ```

4. **Ollama (Local LLM)**: Download and install from https://ollama.ai
   - Run: `ollama serve` in a terminal
   - Pull model: `ollama pull mistral` (or your preferred model)

### Linux Setup

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Fedora
sudo dnf install tesseract-ocr

# Install Ollama: https://ollama.ai
ollama serve
ollama pull mistral
```

### Windows Setup

1. **Tesseract OCR**: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
2. **Ollama**: Download from https://ollama.ai
3. **Python 3.12+**: Download from https://www.python.org/

## Installation

### 1. Clone repository
```bash
git clone <your-repo>
cd ShadowView
```

### 2. Install dependencies with `uv`
```bash
uv sync
```

This installs:
- **Core**: cv2, mss, pytesseract, numpy, PIL, requests, keyboard, pyautogui
- **Dev**: black (formatting), ruff (linting)

If `uv` is not installed, see [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### 3. Verify Ollama is running
```bash
# In another terminal, start Ollama
ollama serve

# Verify model is available
ollama pull mistral
```

## Usage

### Run the Assistant
```bash
uv run python main.py
```

**What happens:**
- Captures your screen every 2 seconds (configurable)
- Detects text changes
- Extracts text via OCR
- Analyzes content type (IQ pattern, math, code, etc.)
- Sends to Ollama for intelligent response
- Displays results with confidence score

### Manual Testing

Run the full pipeline test:
```bash
uv run python test.py
```

Output files are saved to `pipeline_phases_YYYYMMDD_HHMMSS/` with:
- `01_screenshot.png` - Original screen capture
- `02_ocr_result.txt` - Extracted text + confidence
- `03_content_analysis.txt` - Detected content type
- `04_dynamic_prompt.txt` - LLM prompt sent
- `05_ollama_raw_response.txt` - Model response
- `06_formatted_response.txt` - Final formatted output
- `summary.json` - Metadata for all phases

### Single Function Test

Test individual functions in Python:
```bash
uv run python -c "
from main import take_screenshot, extract_text_from_image
img = take_screenshot()
if img:
    text, conf = extract_text_from_image(img)
    print(f'Confidence: {conf:.1f}%')
    print(f'Text: {text[:100]}')
"
```

## Configuration

Edit `CONFIG` in `main.py`:
```python
CONFIG = {
    'ollama_model': 'mistral',        # LLM model (see: ollama list)
    'ollama_host': 'http://localhost:11434',  # Ollama server
    'ocr_languages': 'eng+spa',       # Tesseract languages
    'min_confidence': 40,             # Min OCR confidence (0-100)
    'max_response_length': 300,       # Max response chars
    'temperature': 0.3,               # LLM temperature (0-1)
    'num_predict': 100,               # Max tokens to generate
    'screenshot_interval': 2,         # Seconds between captures
}
```

## Troubleshooting

### "Screenshot failed" / "pyautogui not available"
**macOS:**
```bash
uv sync              # Reinstall dependencies
# Grant Screen Recording permission in System Settings
```

### "No text detected"
- Check if text is visible on screen
- Verify Tesseract is installed: `which tesseract`
- Test OCR manually with `test.py`
- Try adjusting `min_confidence` in CONFIG

### "Ollama not running"
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run assistant
uv run python main.py
```

### "Model not found"
```bash
ollama pull mistral    # Or your preferred model
ollama list           # See available models
```

### Linting & Formatting

Format code:
```bash
uv run black .
```

Check for issues:
```bash
uv run ruff check . --fix
```

## Code Style

This project follows:
- **Type hints**: All functions must have parameter and return type hints
- **Docstrings**: Google-style for all public functions
- **Imports**: Standard lib → Third-party → Local (with blank lines)
- **Naming**: snake_case functions, UPPER_SNAKE_CASE constants, has_/is_/can_ for booleans
- **Error handling**: Catch specific exceptions, use `[ERROR]` prefix for logs
- **Functional style**: Composition over OOP, data pipelines

See [AGENTS.md](AGENTS.md) for detailed guidelines.

## Architecture

```
main.py
├── Screenshot capture (platform-specific)
├── OCR + text extraction
├── Content type analysis
├── Dynamic prompt generation
├── Ollama LLM query
└── Response formatting

test.py
└── Full pipeline testing with phase-by-phase file output
```

## Key Files

- `main.py` - Core application (330 lines)
- `test.py` - Pipeline testing (170 lines)
- `pyproject.toml` - Project metadata & dependencies
- `AGENTS.md` - Coding guidelines for agents
- `README.md` - This file

## Dependencies

| Package | Purpose |
|---------|---------|
| cv2 | Image processing |
| mss | Screen capture (Linux/Windows) |
| pyautogui | Screen capture (macOS) |
| pytesseract | OCR text extraction |
| numpy | Array operations |
| PIL | Image handling |
| requests | HTTP requests to Ollama |
| keyboard | Hotkey support (optional) |
| black | Code formatting |
| ruff | Linting |

## Performance Tips

1. **Reduce screenshot interval**: Change `screenshot_interval` in CONFIG (default: 2s)
2. **Lower OCR confidence threshold**: Adjust `min_confidence` if missing text
3. **Use faster LLM**: Try `ollama pull neural-chat` for faster responses
4. **Disable keyboard hotkeys**: Not essential, can save resources

## Platform Support

| OS | Screenshot | OCR | LLM | Status |
|---|---|---|---|---|
| macOS 12+ | ✅ pyautogui | ✅ tesseract | ✅ Ollama | Full support |
| Linux | ✅ mss | ✅ tesseract | ✅ Ollama | Full support |
| Windows 10+ | ✅ mss | ✅ tesseract | ✅ Ollama | Full support |

## Contributing

For code changes, ensure:
1. Type hints on all functions
2. Google-style docstrings
3. Pass linting: `uv run ruff check .`
4. Format with: `uv run black .`

See [AGENTS.md](AGENTS.md) for detailed guidelines.

## License

MIT

## Support

For issues, see [AGENTS.md](AGENTS.md) troubleshooting section or check GitHub issues.
