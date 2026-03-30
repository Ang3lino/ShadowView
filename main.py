"""
ShadowView AI Assistant
Optimized for macOS with full-screen capture.
"""

import hashlib
import platform
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pytesseract
import requests
from PIL import Image

try:
    import pyautogui
except ImportError:
    pyautogui = None

try:
    from mss import mss
except ImportError:
    mss = None

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None


# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================


def check_dependencies() -> bool:
    """Check required dependencies at startup."""
    errors = []

    required_packages = {
        "pytesseract": pytesseract,
        "cv2": cv2,
        "numpy": np,
        "PIL": Image,
        "requests": requests,
    }
    for name, module in required_packages.items():
        if module is None:
            errors.append(f"Python: {name} not installed (run: uv sync)")

    if platform.system() == "Darwin":
        if pyautogui is None:
            errors.append("macOS: pyautogui not installed (run: uv sync)")
    elif mss is None:
        errors.append("Linux/Windows: mss not installed (run: uv sync)")

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        install_cmd = (
            "brew install tesseract"
            if platform.system() == "Darwin"
            else (
                "sudo apt install tesseract-ocr"
                if platform.system() == "Linux"
                else "Download from https://github.com/UB-Mannheim/tesseract"
            )
        )
        errors.append(f"Tesseract OCR not found ({install_cmd})")

    try:
        resp = requests.get(f"{CONFIG['ollama_host']}/api/tags", timeout=2)
        if resp.status_code != 200:
            raise Exception()
    except Exception:
        errors.append("Ollama not running (run: ollama serve)")

    if errors:
        print("\n" + "=" * 50)
        print("Missing Dependencies")
        print("=" * 50)
        for err in errors:
            print(f"  - {err}")
        print("=" * 50 + "\n")
        return False
    return True


# Configuration
CONFIG = {
    "ollama_model": "mistral",
    "ollama_host": "http://localhost:11434",
    "ocr_languages": "eng+spa",
    "min_confidence": 40,
    "max_response_length": 300,
    "temperature": 0.3,
    "num_predict": 100,
    "screenshot_interval": 2,
    "debug_mode": False,
}

# Global state
_last_screenshot_hash: Optional[str] = None
_capture_region: Optional[Tuple[int, int, int, int]] = None
_region_definition_active = False
_region_start_pos: Optional[Tuple[int, int]] = None
_last_cursor_pos: Optional[Tuple[int, int]] = None
_last_capture_path: Optional[Path] = None
_captures_dir: Path = Path("captures")

# ============================================================================
# SCREEN CAPTURE
# ============================================================================


def normalize_rectangle(
    x1: int, y1: int, x2: int, y2: int
) -> Tuple[int, int, int, int]:
    """Ensure left < right and top < bottom."""
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def clamp_rectangle(
    left: int, top: int, right: int, bottom: int
) -> Tuple[int, int, int, int]:
    """Clamp rectangle to screen bounds."""
    if platform.system() == "Darwin":
        if pyautogui is None:
            return left, top, right, bottom
        screen = pyautogui.screenshot()
        screen_width, screen_height = screen.size
    else:
        try:
            with mss() as sct:
                monitor = sct.monitors[1]
                screen_width, screen_height = monitor["width"], monitor["height"]
        except Exception:
            return left, top, right, bottom

    return (
        max(0, min(left, screen_width)),
        max(0, min(top, screen_height)),
        max(0, min(right, screen_width)),
        max(0, min(bottom, screen_height)),
    )


def wait_for_region_definition() -> Optional[Tuple[int, int, int, int]]:
    """Wait for Ctrl+drag to define capture region. Returns (left, top, right, bottom)."""
    global _region_definition_active, _region_start_pos

    if pynput_keyboard is None:
        print("[ERROR] pynput not available (run: uv sync)")
        return None

    print("\nHold Ctrl, move to first corner, release at opposite corner")
    _region_definition_active = False
    _region_start_pos = None
    ctrl_pressed = False

    def on_press(key):
        nonlocal ctrl_pressed
        try:
            if key == pynput_keyboard.Key.ctrl_l or key == pynput_keyboard.Key.ctrl_r:
                if not ctrl_pressed:
                    global _region_definition_active, _region_start_pos
                    _region_definition_active = True
                    _region_start_pos = pyautogui.position()
                    ctrl_pressed = True
        except AttributeError:
            pass

    def on_release(key):
        nonlocal ctrl_pressed
        try:
            if key == pynput_keyboard.Key.ctrl_l or key == pynput_keyboard.Key.ctrl_r:
                if ctrl_pressed:
                    global _region_definition_active
                    _region_definition_active = False
                    ctrl_pressed = False
                    return False
        except AttributeError:
            pass

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    while not _region_start_pos:
        time.sleep(0.1)
    while _region_definition_active:
        time.sleep(0.1)

    end_pos = pyautogui.position()
    listener.stop()

    if _region_start_pos == end_pos:
        print("  Mouse didn't move. Region unchanged.")
        return None

    left, top, right, bottom = normalize_rectangle(
        _region_start_pos[0], _region_start_pos[1], end_pos[0], end_pos[1]
    )
    left, top, right, bottom = clamp_rectangle(left, top, right, bottom)
    print(
        f"  Region: ({left}, {top}) to ({right}, {bottom}) | {right - left}x{bottom - top}"
    )
    return left, top, right, bottom


def take_screenshot() -> Optional[Image.Image]:
    """Capture the defined region."""
    global _capture_region

    if _capture_region is None:
        print("[ERROR] Capture region not defined")
        return None

    left, top, right, bottom = _capture_region

    if platform.system() == "Darwin":
        if pyautogui is None:
            print("[ERROR] pyautogui not available")
            return None
        try:
            return pyautogui.screenshot().crop((left, top, right, bottom))
        except Exception as e:
            print(f"[ERROR] Screenshot: {e}")
            return None
    else:
        if mss is None:
            print("[ERROR] mss not available")
            return None
        try:
            with mss() as sct:
                screenshot = sct.grab(
                    {
                        "left": left,
                        "top": top,
                        "width": right - left,
                        "height": bottom - top,
                    }
                )
                return Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        except Exception as e:
            print(f"[ERROR] Screenshot: {e}")
            return None


def has_screen_changed(image: Image.Image) -> bool:
    """Check if screen content changed by comparing with last saved image."""
    global _last_capture_path
    current_hash = hashlib.md5(image.tobytes()).hexdigest()

    if _last_capture_path and _last_capture_path.exists():
        last_image = Image.open(_last_capture_path)
        last_hash = hashlib.md5(last_image.tobytes()).hexdigest()
        if current_hash == last_hash:
            return False

    _last_screenshot_hash = current_hash
    return True


def save_capture(image: Image.Image) -> Path:
    """Save capture to captures/ directory with timestamp filename."""
    global _last_capture_path
    _captures_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    milliseconds = int(time.time() * 1000) % 1000
    filename = f"capture_{timestamp}_{milliseconds:03d}.png"
    filepath = _captures_dir / filename
    image.save(filepath)
    _last_capture_path = filepath
    return filepath


# ============================================================================
# OCR
# ============================================================================


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for better OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def extract_text_from_image(image: Image.Image) -> Tuple[str, float]:
    """Extract text with confidence score."""
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        ocr_data = pytesseract.image_to_data(
            preprocess_image(img_cv),
            lang=CONFIG["ocr_languages"],
            output_type=pytesseract.Output.DICT,
        )
        texts, confidences = [], []
        for text, conf in zip(ocr_data["text"], ocr_data["conf"]):
            if text.strip() and int(conf) > CONFIG["min_confidence"]:
                texts.append(text)
                confidences.append(int(conf))
        return " ".join(texts), np.mean(confidences) if confidences else 0.0
    except Exception as e:
        print(f"[ERROR] OCR: {e}")
        return "", 0.0


# ============================================================================
# LLM
# ============================================================================
def analyze_content(text: str) -> str:
    """Detect content type."""
    text_lower = text.lower()
    patterns = {
        "iq_pattern": ["pattern", "sequence", "next", "complete", "matrix"],
        "iq_math": ["number", "equation", "calculate", "solve", "math"],
        "iq_spatial": ["shape", "rotate", "mirror", "spatial", "figure"],
        "iq_verbal": ["word", "analogy", "meaning", "verbal", "language"],
        "code": ["def ", "class ", "import ", "function"],
    }
    for content_type, keywords in patterns.items():
        if any(kw in text_lower for kw in keywords):
            return content_type
    return "general"


def build_dynamic_prompt(text: str, content_type: str) -> str:
    """Build LLM prompt."""
    base = """You are an IQ test assistant. Respond in MAX 2 LINES.
Be direct and concise. Use bullets if helpful.
If unsure, start with [ALERTA]."""

    instructions = {
        "iq_pattern": "Pattern detected. Explain logic and give answer.",
        "iq_math": "Math problem. Provide solution.",
        "iq_spatial": "Spatial problem. Describe transformation.",
        "iq_verbal": "Verbal problem. Explain relationship.",
        "code": "Code snippet. Explain briefly.",
        "general": "Analyze and respond concisely.",
    }

    return f"""{base}
{instructions.get(content_type, instructions["general"])}

Content:
{text[:500]}

Answer (max 2 lines):"""


def query_ollama(prompt: str) -> Optional[str]:
    """Query Ollama."""
    try:
        response = requests.post(
            f"{CONFIG['ollama_host']}/api/generate",
            json={
                "model": CONFIG["ollama_model"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": CONFIG["temperature"],
                    "num_predict": CONFIG["num_predict"],
                },
            },
            timeout=10,
        )
        return (
            response.json().get("response", "").strip()
            if response.status_code == 200
            else None
        )
    except Exception as e:
        print(f"[ERROR] Ollama: {e}")
        return None


def format_response(response: str) -> str:
    """Format response."""
    if not response:
        return "[ALERTA] No response"
    if len(response) > CONFIG["max_response_length"]:
        response = response[: CONFIG["max_response_length"]] + "..."
    if any(
        kw in response.lower()
        for kw in ["not sure", "maybe", "perhaps", "could be", "uncertain"]
    ) and not response.startswith("[ALERTA"):
        response = f"[ALERTA] {response}"
    return response


# ============================================================================
# MAIN LOOP
# ============================================================================
def process_screen() -> Tuple[str, str, float]:
    """Full pipeline."""
    image: Optional[Image.Image] = take_screenshot()
    if not image:
        return "", "[ERROR] Screenshot failed", 0.0
    extracted_text: str = ""
    confidence: float = 0.0
    extracted_text, confidence = extract_text_from_image(image)
    if not extracted_text:
        return "", "[INFO] No text detected", confidence
    content_type = analyze_content(extracted_text)
    prompt = build_dynamic_prompt(extracted_text, content_type)
    raw_response: Optional[str] = query_ollama(prompt)
    if raw_response:
        response_text = format_response(raw_response)
    else:
        response_text = "[ERROR] No response"
    return extracted_text, response_text, confidence


def check_cursor_moved() -> bool:
    """Check if cursor moved since last capture."""
    global _last_cursor_pos
    current_pos = pyautogui.position()
    moved = _last_cursor_pos is not None and current_pos != _last_cursor_pos
    _last_cursor_pos = current_pos
    return moved


def setup_region_listener(on_ctrl_callback) -> None:
    """Setup listener to redefine region on Ctrl press."""
    if pynput_keyboard is None:
        return
    ctrl_pressed = False

    def on_press(key):
        nonlocal ctrl_pressed
        try:
            if key in (pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r):
                if not ctrl_pressed:
                    ctrl_pressed = True
                    on_ctrl_callback()
        except AttributeError:
            pass

    def on_release(key):
        nonlocal ctrl_pressed
        try:
            if key in (pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r):
                ctrl_pressed = False
        except AttributeError:
            pass

    pynput_keyboard.Listener(on_press=on_press, on_release=on_release).start()


def run_assistant() -> None:
    """Main loop."""
    global _capture_region
    print("\n" + "=" * 50)
    print("ShadowView AI Assistant")
    print("=" * 50)
    if not check_dependencies():
        return
    if platform.system() == "Darwin" and pyautogui is not None:
        try:
            test = pyautogui.screenshot()
            print(f"Screenshot working: {test.size}")
        except Exception as e:
            print(f"Screenshot permission issue: {e}")
            print("Configure: System Settings -> Privacy -> Screen Recording")
            return
    if pynput_keyboard is None:
        print("pynput not available (run: uv sync)")
        return

    print(
        f"\nConfig: {CONFIG['screenshot_interval']}s interval | OCR: {CONFIG['ocr_languages']} | Model: {CONFIG['ollama_model']}"
    )
    print("Controls: Ctrl to define region | Ctrl+C to stop")

    print("\n" + "=" * 50)
    _capture_region = wait_for_region_definition()
    if _capture_region is None:
        print("Failed to define region")
        return
    print("=" * 50)

    def on_ctrl_in_loop():
        global _capture_region
        print("\nRedefining region...")
        new_region = wait_for_region_definition()
        if new_region:
            _capture_region = new_region
            print("Region updated")

    setup_region_listener(on_ctrl_in_loop)

    try:
        while True:
            if not check_cursor_moved():
                time.sleep(CONFIG["screenshot_interval"])
                continue
            current_image = take_screenshot()
            if current_image:
                filepath = save_capture(current_image)
                if has_screen_changed(current_image):
                    print(f"\nScreen changed... ({filepath.name})")
                question, response, confidence = process_screen()
                if question:
                    print(f" Q ({confidence:.0f}%): {question[:200]}...")
                    print(f" A: {response}")
                else:
                    print("  No text detected")
            time.sleep(CONFIG["screenshot_interval"])
    except KeyboardInterrupt:
        print("\n\nAssistant stopped")


if __name__ == "__main__":
    run_assistant()
