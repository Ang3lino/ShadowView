# ShadowView Agent Guidelines

Guidelines for agentic coding assistants working on the ShadowView project.

## Build, Test & Run Commands

### Using `uv` (preferred)

- **Install dependencies**: `uv sync`
- **Run main application**: `uv run python main.py`
- **Run tests/pipeline**: `uv run python test.py`
- **Manual single capture**: `uv run python main.py manual`
- **Lint checks**: `uv run ruff check .`
- **Format code**: `uv run black .`
- **Fix linting issues**: `uv run ruff check . --fix && uv run black .`

### Running a Single Test
This project runs pipeline phases sequentially in `test.py`. To test individual functions:

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

## Code Style Guidelines

### Imports
- **Order**: Standard library → Third-party → Local modules (with blank lines between)
- **Style**: Use explicit imports, not wildcards
- **Location**: All imports at module top, after module docstring
- **Grouping**: Standard library, then third-party, then local (see main.py lines 6-16)

Example:
```python
"""Module docstring."""

import hashlib
import time

import cv2
import numpy as np
import requests
from PIL import Image

from mymodule import helper
```

### Formatting
- **Line length**: No strict limit, but keep readable
- **Indentation**: 4 spaces (enforced by Black)
- **Tool**: Black handles all formatting (run `uv run black .`)
- **Tool**: Ruff for linting (run `uv run ruff check . --fix`)

### Type Hints
- **Required**: All function parameters and return types must have hints
- **Optional types**: Use `Optional[T]` for nullable values
- **Tuples**: Use `Tuple[type, ...]` for specific tuple signatures
- **Imports**: Use `from typing import Optional, Tuple`

Example:
```python
def extract_text_from_image(
    image: Image.Image, min_confidence: int = None
) -> Tuple[str, float]:
    """Extract text with confidence score."""
```

### Naming Conventions
- **Functions**: `snake_case` (e.g., `take_screenshot`, `extract_text_from_image`)
- **Classes**: `PascalCase` (rarely used; prefer functions)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `CONFIG` is intentional global state)
- **Private functions**: Prefix with `_` (e.g., `_last_screenshot_hash`)
- **Booleans**: Start with `has_`, `is_`, `can_` prefix (e.g., `has_screen_changed`)

### Docstrings
- **Style**: Google-style with triple quotes
- **Required**: All public functions must have docstrings
- **Format**: Description, then Args/Returns sections
- **Length**: Keep descriptions concise (1-2 sentences)

Example:
```python
def analyze_content(text: str) -> str:
    """
    Dynamically detect content type from text.

    Args:
        text: Extracted text to analyze.

    Returns:
        Content type: 'iq_pattern', 'iq_math', 'code', 'general', etc.
    """
```

### Error Handling
- **Pattern**: Catch specific exceptions, not bare `except`
- **Logging**: Use print with `[ERROR]` prefix for errors
- **Fallback**: Return sensible defaults (None, empty string, False)
- **Messages**: Include context (e.g., `[ERROR] Screenshot: {e}`)

Example:
```python
try:
    result = do_something()
except SpecificException as e:
    print(f"[ERROR] Operation: {e}")
    return None
```

### Code Organization
- **Sections**: Use `# ============...` headers for logical sections
- **Structure**: Follow main.py pattern:
  1. Module docstring & imports
  2. CONFIG/constants
  3. Utility functions (by category)
  4. Main orchestration functions
  5. CLI interface (`if __name__ == "__main__"`)
- **Comments**: Use for "why" not "what"; code is self-documenting
- **Global state**: Minimize; use private globals (`_var_name`) when necessary

### Functional Style
- **Preference**: Functional composition over OOP
- **Return values**: Functions return data, not side effects
- **Pipelines**: Build data pipelines with function composition (see `process_screen()`)
- **Constants**: Store config in module-level `CONFIG` dict (main.py:19-29)

### Project Dependencies
- **Core**: cv2 (OpenCV), mss, pytesseract, numpy, PIL, requests
- **Dev**: black (formatting), ruff (linting)
- **Python**: 3.12+ (from pyproject.toml)
- **External services**: Ollama (local LLM server on localhost:11434)

### Special Conventions
- **Alert syntax**: Use `[ALERTA]` for uncertain responses (Spanish convention)
- **Status logging**: Use emoji prefixes for clarity (📸, 💡, ✅, ⚠️, 🔄)
- **Configuration**: Never hardcode; use CONFIG dict for all settings
- **Change detection**: Use hash comparison for efficiency (main.py:59-77)

## Testing Patterns

- **Pipeline testing**: Run `uv run python test.py` to exercise all phases
- **Phases**: Screenshot → OCR → Content Analysis → Prompt → Ollama → Format
- **Output**: Results saved to `pipeline_results.json` after test
- **Integration**: Manual testing via `uv run python main.py manual`

## Key Files
- `main.py` (399 lines): Core application with screen capture, OCR, LLM integration
- `test.py` (121 lines): Pipeline testing with phase-by-phase output
- `reference.py`: Backup/reference copy of main.py
- `pyproject.toml`: Project metadata and dependencies
