"""
IQ-Edge: Intelligent Interview Assistant
Versión optimizada para macOS con captura de pantalla completa.
"""

import platform
import time
import hashlib
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
}

# Estado global
_last_screenshot_hash = None

# ============================================================================
# SCREEN CAPTURE (macOS Compatible)
# ============================================================================


def take_screenshot() -> Optional[Image.Image]:
    """
    Captura pantalla completa compatible con macOS.
    En macOS usa pyautogui (requiere permisos de Acceso a Pantalla).
    Fallback a mss en Linux/Windows.
    """
    # macOS
    if platform.system() == "Darwin":
        if pyautogui is None:
            print("[ERROR] pyautogui not available on macOS")
            print("   Install with: uv sync")
            return None

        try:
            screenshot = pyautogui.screenshot()
            return screenshot
        except Exception as e:
            print(f"[ERROR] Screenshot macOS: {e}")
            print("   Verifica permisos: Privacidad y Seguridad -> Acceso a Pantalla")
            return None

    # Linux/Windows
    else:
        if mss is None:
            print("[ERROR] mss not available")
            print("   Install with: uv sync")
            return None

        try:
            with mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
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


def run_assistant() -> None:
    """
    Loop principal del asistente.
    Monitorea cambios de pantalla y procesa contenido.
    """
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

    print("\n⚙️  Configuration:")
    print(f"   - Screenshot every {CONFIG['screenshot_interval']}s")
    print(f"   - OCR: {CONFIG['ocr_languages']}")
    print(f"   - Model: {CONFIG['ollama_model']}")
    print("\n⌨️  Controls:")
    print("   - Press Ctrl+C to stop")

    print("\n" + "=" * 50)

    try:
        while True:
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
