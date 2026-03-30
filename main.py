"""
IQ-Edge: Intelligent Interview Assistant
Versión optimizada para macOS con captura de pantalla completa.
"""

import platform
import time
import hashlib
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import pytesseract
import requests
from PIL import Image

# Import platform-specific and optional modules
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

# Configuración
CONFIG = {
    "ollama_model": "mistral",
    "ollama_host": "http://localhost:11434",
    "ocr_languages": "eng+spa",
    "min_confidence": 40,
    "max_response_length": 300,
    "temperature": 0.3,
    "num_predict": 100,
    "screenshot_interval": 2,
    "debug_mode": False,  # Cambiar a True para ver coordenadas de captura
}

# Estado global
_last_screenshot_hash = None
_capture_region: Optional[Tuple[int, int, int, int]] = (
    None  # (left, top, right, bottom)
)
_region_definition_active = False
_region_start_pos: Optional[Tuple[int, int]] = None
_last_cursor_pos: Optional[Tuple[int, int]] = None

# ============================================================================
# SCREEN CAPTURE (macOS Compatible)
# ============================================================================


def normalize_rectangle(
    x1: int, y1: int, x2: int, y2: int
) -> Tuple[int, int, int, int]:
    """
    Normaliza un rectángulo para asegurar left < right y top < bottom.

    Args:
        x1, y1, x2, y2: Dos esquinas opuestas del rectángulo.

    Returns:
        Tuple de (left, top, right, bottom).
    """
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    return left, top, right, bottom


def clamp_rectangle(
    left: int, top: int, right: int, bottom: int
) -> Tuple[int, int, int, int]:
    """
    Ajusta un rectángulo para que no rebase los límites de la pantalla.

    Args:
        left, top, right, bottom: Coordenadas del rectángulo.

    Returns:
        Rectángulo ajustado dentro de los límites de pantalla.
    """
    if platform.system() == "Darwin":
        if pyautogui is None:
            return left, top, right, bottom
        screen = pyautogui.screenshot()
        screen_width, screen_height = screen.size
    else:
        try:
            with mss() as sct:
                monitor = sct.monitors[1]
                screen_width = monitor["width"]
                screen_height = monitor["height"]
        except Exception:
            return left, top, right, bottom

    left = max(0, min(left, screen_width))
    right = max(0, min(right, screen_width))
    top = max(0, min(top, screen_height))
    bottom = max(0, min(bottom, screen_height))

    return left, top, right, bottom


def wait_for_region_definition() -> Optional[Tuple[int, int, int, int]]:
    """
    Espera a que el usuario presione Ctrl y defina un rectángulo.
    Retorna las esquinas opuestas del rectángulo.

    Returns:
        Tuple de (left, top, right, bottom) o None si hay error.
    """
    global _region_definition_active, _region_start_pos, _last_cursor_pos

    if pynput_keyboard is None:
        print("[ERROR] pynput keyboard library not available")
        print("   Install with: uv sync")
        return None

    print("\n⌨️  Press and hold Ctrl to define capture region")
    print("   Move cursor to first corner, then release Ctrl at opposite corner")
    print("   (Press Ctrl again anytime to redefine region)")

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
                    if CONFIG["debug_mode"]:
                        print(f"   [DEBUG] Ctrl pressed at {_region_start_pos}")
        except AttributeError:
            pass

    def on_release(key):
        nonlocal ctrl_pressed
        try:
            if key == pynput_keyboard.Key.ctrl_l or key == pynput_keyboard.Key.ctrl_r:
                if ctrl_pressed:
                    global _region_definition_active
                    end_pos = pyautogui.position()
                    if CONFIG["debug_mode"]:
                        print(f"   [DEBUG] Ctrl released at {end_pos}")
                    _region_definition_active = False
                    ctrl_pressed = False
                    return False  # Stop listener
        except AttributeError:
            pass

    # Crear listener
    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Esperar a que se complete la definición
    while not _region_start_pos:
        time.sleep(0.1)

    while _region_definition_active:
        time.sleep(0.1)

    # Obtener posición final
    end_pos = pyautogui.position()

    # Detener listener
    listener.stop()

    if _region_start_pos == end_pos:
        print("   ⚠️  Mouse didn't move. Region not changed.")
        return None

    left, top, right, bottom = normalize_rectangle(
        _region_start_pos[0], _region_start_pos[1], end_pos[0], end_pos[1]
    )
    left, top, right, bottom = clamp_rectangle(left, top, right, bottom)

    region_width = right - left
    region_height = bottom - top
    print(f"   ✅ Region defined: ({left}, {top}) to ({right}, {bottom})")
    print(f"   📐 Size: {region_width}x{region_height}")

    return left, top, right, bottom


def take_screenshot() -> Optional[Image.Image]:
    """
    Captura la región personalizada definida por el usuario.
    Requiere que _capture_region esté definida.
    """
    global _capture_region

    if _capture_region is None:
        print("[ERROR] Capture region not defined")
        return None

    left, top, right, bottom = _capture_region

    # macOS
    if platform.system() == "Darwin":
        if pyautogui is None:
            print("[ERROR] pyautogui not available on macOS")
            return None

        try:
            full_screenshot = pyautogui.screenshot()
            cropped = full_screenshot.crop((left, top, right, bottom))
            return cropped
        except Exception as e:
            print(f"[ERROR] Screenshot macOS: {e}")
            return None

    # Linux/Windows
    else:
        if mss is None:
            print("[ERROR] mss not available")
            return None

        try:
            with mss() as sct:
                region = {
                    "left": left,
                    "top": top,
                    "width": right - left,
                    "height": bottom - top,
                }
                screenshot = sct.grab(region)
                return Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        except Exception as e:
            print(f"[ERROR] Screenshot: {e}")
            return None


def has_screen_changed(image: Image.Image) -> bool:
    """Detectar si la pantalla cambió."""
    global _last_screenshot_hash

    current_hash = hashlib.md5(image.tobytes()).hexdigest()

    if _last_screenshot_hash is None:
        _last_screenshot_hash = current_hash
        return True

    if current_hash != _last_screenshot_hash:
        _last_screenshot_hash = current_hash
        return True

    return False


# ============================================================================
# OCR FUNCTIONS
# ============================================================================


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocesar imagen para mejor OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return cv2.medianBlur(thresh, 3)


def extract_text_from_image(image: Image.Image) -> Tuple[str, float]:
    """Extraer texto de imagen con confianza."""
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed = preprocess_image(img_cv)

        ocr_data = pytesseract.image_to_data(
            processed, lang=CONFIG["ocr_languages"], output_type=pytesseract.Output.DICT
        )

        texts = []
        confidences = []

        for text, conf in zip(ocr_data["text"], ocr_data["conf"]):
            if text.strip() and int(conf) > CONFIG["min_confidence"]:
                texts.append(text)
                confidences.append(int(conf))

        extracted = " ".join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return extracted, avg_confidence

    except Exception as e:
        print(f"[ERROR] OCR: {e}")
        return "", 0.0


# ============================================================================
# LLM FUNCTIONS
# ============================================================================


def analyze_content(text: str) -> str:
    """Detectar tipo de contenido."""
    text_lower = text.lower()

    patterns = {
        "iq_pattern": ["pattern", "sequence", "next", "complete", "matrix"],
        "iq_math": ["number", "equation", "calculate", "solve", "math"],
        "iq_spatial": ["shape", "rotate", "mirror", "spatial", "figure"],
        "iq_verbal": ["word", "analogy", "meaning", "verbal", "language"],
        "code": ["def ", "class ", "import ", "function"],
    }

    for content_type, keywords in patterns.items():
        if any(keyword in text_lower for keyword in keywords):
            return content_type

    return "general"


def build_dynamic_prompt(text: str, content_type: str) -> str:
    """Construir prompt para LLM."""
    base = """Eres un asistente para test de IQ. Responde en MÁXIMO 2 LÍNEAS.
Sé directo, conciso. Usa viñetas si ayuda.
Si no estás seguro, comienza con [ALERTA]."""

    instructions = {
        "iq_pattern": "Es un patrón. Da la lógica y la respuesta.",
        "iq_math": "Es un problema matemático. Da la solución.",
        "iq_spatial": "Es un problema espacial. Describe la transformación.",
        "iq_verbal": "Es un problema verbal. Da la relación.",
        "code": "Es código. Explica brevemente.",
        "general": "Analiza y responde concisamente.",
    }

    instruction = instructions.get(content_type, instructions["general"])

    return f"""{base}
{instruction}

Contenido:
{text[:500]}

Respuesta (máximo 2 líneas):"""


def query_ollama(prompt: str) -> Optional[str]:
    """Consultar Ollama."""
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

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        return None

    except Exception as e:
        print(f"[ERROR] Ollama: {e}")
        return None


def format_response(response: str) -> str:
    """Formatear respuesta."""
    if not response:
        return "[ALERTA] No response"

    if len(response) > CONFIG["max_response_length"]:
        response = response[: CONFIG["max_response_length"]] + "..."

    uncertain = ["no estoy seguro", "quizás", "tal vez", "podría ser"]
    if any(kw in response.lower() for kw in uncertain):
        if not response.startswith("[ALERTA"):
            response = f"[ALERTA] {response}"

    return response


# ============================================================================
# MAIN LOOP
# ============================================================================


def process_screen() -> Tuple[str, str, float]:
    """Pipeline completo."""
    image = take_screenshot()
    if not image:
        return "", "[ERROR] Screenshot failed", 0.0

    text, confidence = extract_text_from_image(image)

    if not text:
        return "", "[INFO] No text detected", confidence

    content_type = analyze_content(text)
    prompt = build_dynamic_prompt(text, content_type)
    raw_response = query_ollama(prompt)
    response = format_response(raw_response) if raw_response else "[ERROR] No response"

    return text, response, confidence


def check_cursor_moved() -> bool:
    """Verifica si el cursor se movió desde la última captura."""
    global _last_cursor_pos

    current_pos = pyautogui.position()

    if _last_cursor_pos is None:
        _last_cursor_pos = current_pos
        return False

    moved = current_pos != _last_cursor_pos
    _last_cursor_pos = current_pos

    return moved


def setup_region_listener(on_ctrl_callback) -> None:
    """Configura listener para redefinir región cuando Ctrl se presione."""
    if pynput_keyboard is None:
        return

    ctrl_pressed = False

    def on_press(key):
        nonlocal ctrl_pressed
        try:
            if key == pynput_keyboard.Key.ctrl_l or key == pynput_keyboard.Key.ctrl_r:
                if not ctrl_pressed:
                    ctrl_pressed = True
                    on_ctrl_callback()
        except AttributeError:
            pass

    def on_release(key):
        nonlocal ctrl_pressed
        try:
            if key == pynput_keyboard.Key.ctrl_l or key == pynput_keyboard.Key.ctrl_r:
                ctrl_pressed = False
        except AttributeError:
            pass

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()


def run_assistant() -> None:
    """
    Loop principal del asistente.
    Permite definir región con Ctrl y monitorea cambios de pantalla.
    """
    global _capture_region

    print("\n" + "=" * 50)
    print("IQ-Edge Interview Assistant (macOS)")
    print("=" * 50)

    # Verificar permisos en macOS
    if platform.system() == "Darwin":
        if pyautogui is None:
            print("\n⚠️  pyautogui not available")
            print("   Run: uv sync")
            return

        try:
            test = pyautogui.screenshot()
            print(f"✅ Screenshot working: {test.size}")
        except Exception as e:
            print(f"⚠️  Screenshot permission issue: {e}")
            print("   Configure: System Settings -> Privacy -> Screen Recording")
            return

    if pynput_keyboard is None:
        print("\n⚠️  pynput keyboard library not available")
        print("   Run: uv sync")
        return

    print("\n⚙️  Configuration:")
    print(f"   - Screenshot every {CONFIG['screenshot_interval']}s")
    print(f"   - OCR: {CONFIG['ocr_languages']}")
    print(f"   - Model: {CONFIG['ollama_model']}")
    print("\n⌨️  Controls:")
    print("   - Press Ctrl to define/redefine capture region")
    print("   - Press Ctrl+C to stop")

    # Definir región inicial
    print("\n" + "=" * 50)
    _capture_region = wait_for_region_definition()

    if _capture_region is None:
        print("\n❌ Failed to define region")
        return

    print("\n" + "=" * 50)

    def on_ctrl_pressed():
        """Callback cuando Ctrl es presionado durante el loop."""
        global _capture_region
        print("\n🔄 Redefining region...")
        new_region = wait_for_region_definition()
        if new_region is not None:
            _capture_region = new_region
            print("   ✅ Region updated")
        else:
            print("   ⚠️  Region unchanged")

    setup_region_listener(on_ctrl_pressed)

    try:
        while True:
            # Verificar si el cursor se movió
            if not check_cursor_moved():
                time.sleep(CONFIG["screenshot_interval"])
                continue

            current_image = take_screenshot()
            if current_image and has_screen_changed(current_image):
                print("\n🔄 Screen changed...")
                question, response, confidence = process_screen()

                if question:
                    print(f"📝 Q ({confidence:.0f}%): {question[:100]}...")
                    print(f"💡 A: {response}")
                else:
                    print("⏸️  No text detected")

            time.sleep(CONFIG["screenshot_interval"])

    except KeyboardInterrupt:
        print("\n\n👋 Assistant stopped")


if __name__ == "__main__":
    run_assistant()
