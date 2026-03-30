# ShadowView AI Assistant

An intelligent assistant that monitors your screen, extracts text via OCR, analyzes content type, and provides AI-powered responses using a local LLM (Ollama).

## Features

- 📸 **Full-screen capture** compatible with macOS, Linux, and Windows
- 🔍 **OCR text extraction** with confidence scoring (English + Spanish)
- 🧠 **Content analysis** - Detects IQ patterns, math problems, spatial/verbal questions, code
- 💡 **Dynamic prompts** - Tailored LLM instructions based on content type
- 🎯 **Local LLM integration** - Uses Ollama for privacy and speed
- 🔄 **Change detection** - Only processes when screen content changes

## Prerequisites

### Required Software

| Software | Purpose | Install |
|----------|---------|---------|
| Python 3.12+ | Runtime | [python.org](https://www.python.org/) |
| Tesseract OCR | Text extraction | See below |
| Ollama | Local LLM | [ollama.ai](https://ollama.ai) |

### macOS

```bash
brew install tesseract
# Download Ollama from https://ollama.ai
```

### Linux

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# Fedora
sudo dnf install tesseract-ocr

# Download Ollama from https://ollama.ai
```

### Windows

1. Download Tesseract from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Download Ollama from [ollama.ai](https://ollama.ai)

### macOS: Screen Recording Permission

Go to: **System Settings → Privacy & Security → Screen Recording**  
Add Terminal or your IDE to the allowed apps.

## Installation

```bash
git clone <your-repo>
cd ShadowView
uv sync
```

## Quick Start

### 1. Start Ollama (in another terminal)

```bash
ollama serve
ollama pull mistral
```

### 2. Run the assistant

```bash
uv run python main.py
```

**What happens:**
- Captures your screen every 2 seconds
- Detects text changes
- Extracts text via OCR
- Analyzes content type
- Sends to Ollama for intelligent response

## Testing

Run the full pipeline test:

```bash
uv run python test.py
```

Test a single capture:

```bash
uv run python -c "
from main import take_screenshot, extract_text_from_image
img = take_screenshot()
if img:
    text, conf = extract_text_from_image(img)
    print(f'Confidence: {conf:.1f}%')
"
```

## Configuration

Edit `CONFIG` in `main.py`:

```python
CONFIG = {
    'ollama_model': 'mistral',        # LLM model
    'ollama_host': 'http://localhost:11434',
    'ocr_languages': 'eng+spa',       # Tesseract languages
    'min_confidence': 40,             # Min OCR confidence (0-100)
    'screenshot_interval': 2,         # Seconds between captures
}
```

## Troubleshooting

### "tesseract is not installed or it's not in your PATH"
```bash
brew install tesseract  # macOS
sudo apt install tesseract-ocr  # Ubuntu
```

### "Ollama not running"
```bash
ollama serve
ollama pull mistral
```

### "No text detected"
- Check if text is visible on screen
- Verify Tesseract: `which tesseract`
- Lower `min_confidence` in CONFIG

## Linting & Formatting

```bash
uv run ruff check . --fix
uv run black .
```

## Dependencies

| Package | Purpose |
|---------|---------|
| cv2, numpy, PIL | Image processing |
| mss, pyautogui | Screen capture |
| pytesseract | OCR text extraction |
| requests | HTTP to Ollama |
| pynput | Hotkey support |

## License

MIT
