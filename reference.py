"""
IQ-Edge: Intelligent Interview Assistant
Pythonic implementation with functional style and minimal abstraction.
"""

import hashlib
import time
import cv2
import numpy as np
import pytesseract
import requests

from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image
from mss import mss


CONFIG = {
    "ollama_model": "mistral",
    "ollama_host": "http://localhost:11434",
    "primary_monitor": 1,
    "ocr_languages": "eng+spa",
    "min_confidence": 40,
    "max_response_length": 300,
    "temperature": 0.3,
    "num_predict": 100,
    "screenshot_interval": 2,
}

# Global state (minimal, just for change detection)
_last_screenshot_hash = None
_sct = mss()


# ============================================================================
# SCREEN CAPTURE FUNCTIONS
# ============================================================================
def take_screenshot(monitor: int = None) -> Optional[Image.Image]:
    """
    Capture screenshot of specified monitor.

    Args:
        monitor: Monitor number (1 = primary). Uses CONFIG if None.

    Returns:
        PIL Image or None if failed.
    """
    try:
        monitor = monitor or CONFIG["primary_monitor"]
        monitor_config = _sct.monitors[monitor]
        screenshot = _sct.grab(monitor_config)
        return Image.frombytes("RGB", screenshot.size, screenshot.rgb)
    except Exception as e:
        print(f"[ERROR] Screenshot: {e}")
        return None


def has_screen_changed(image: Image.Image) -> bool:
    """
    Detect if screen has changed using hash comparison.

    Args:
        image: Current screenshot.

    Returns:
        True if screen changed significantly.
    """
    global _last_screenshot_hash
    current_hash = hashlib.md5(image.tobytes()).hexdigest()
    if _last_screenshot_hash is None:
        _last_screenshot_hash = current_hash
        return True
    if current_hash != _last_screenshot_hash:
        _last_screenshot_hash = current_hash
        return True
    return False


# OCR FUNCTIONS
# ============================================================================
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better OCR results.

    Args:
        image: OpenCV image (BGR).

    Returns:
        Processed image (binary threshold).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return cv2.medianBlur(thresh, 3)


def extract_text_from_image(
    image: Image.Image, min_confidence: int = None
) -> Tuple[str, float]:
    """
    Extract text from PIL Image with confidence score.

    Args:
        image: PIL Image to process.
        min_confidence: Minimum confidence threshold (0-100).

    Returns:
        Tuple of (extracted_text, average_confidence).
    """
    min_confidence = min_confidence or CONFIG["min_confidence"]

    try:
        # Convert to OpenCV and preprocess
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed = preprocess_image(img_cv)

        # OCR with confidence data
        ocr_data = pytesseract.image_to_data(
            processed, lang=CONFIG["ocr_languages"], output_type=pytesseract.Output.DICT
        )

        # Filter by confidence
        texts = []
        confidences = []

        for text, conf in zip(ocr_data["text"], ocr_data["conf"]):
            if text.strip() and int(conf) > min_confidence:
                texts.append(text)
                confidences.append(int(conf))

        extracted = " ".join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return extracted, avg_confidence

    except Exception as e:
        print(f"[ERROR] OCR: {e}")
        return "", 0.0


# CONTENT ANALYSIS FUNCTIONS
# ============================================================================
def analyze_content(text: str) -> str:
    """
    Dynamically detect content type from text.

    Args:
        text: Extracted text to analyze.

    Returns:
        Content type: 'iq_pattern', 'iq_math', 'code', 'general', etc.
    """
    text_lower = text.lower()
    # Pattern detection (simplified, but effective)
    patterns = {
        "iq_pattern": ["pattern", "sequence", "next", "complete", "matrix", "series"],
        "iq_math": [
            "number",
            "equation",
            "calculate",
            "solve",
            "math",
            "sum",
            "average",
        ],
        "iq_spatial": ["shape", "rotate", "mirror", "spatial", "figure", "diagram"],
        "iq_verbal": ["word", "analogy", "meaning", "verbal", "language", "synonym"],
        "code": ["def ", "class ", "import ", "function", "return", "print"],
    }
    for content_type, keywords in patterns.items():
        if any(keyword in text_lower for keyword in keywords):
            return content_type
    return "general"


def build_dynamic_prompt(text: str, content_type: str) -> str:
    """
    Build prompt based on detected content type.

    Args:
        text: Extracted text.
        content_type: Type of content detected.

    Returns:
        Formatted prompt for LLM.
    """
    # Base instruction (always the same)
    base = """Eres un asistente para test de IQ. Responde en MÁXIMO 2 LÍNEAS.
Sé directo, conciso. Usa viñetas si ayuda.
Si no estás seguro, comienza con [ALERTA]."""

    # Type-specific instruction
    type_instructions = {
        "iq_pattern": "Es un problema de patrones/secuencias. Explica la lógica del patrón y da la respuesta.",
        "iq_math": "Es un problema matemático/lógico. Da la solución con breve explicación.",
        "iq_spatial": "Es un problema espacial/visual. Describe la transformación y da la respuesta.",
        "iq_verbal": "Es un problema verbal/analogías. Da la relación y la respuesta.",
        "code": "Es código. Explica brevemente qué hace o señala posibles errores.",
        "general": "Analiza el contenido y responde de forma útil pero concisa.",
    }

    instruction = type_instructions.get(content_type, type_instructions["general"])

    return f"""{base}
{instruction}

Contenido:
{text[:500]}

Respuesta (máximo 2 líneas):"""


# LLM FUNCTIONS
# ============================================================================


def query_ollama(prompt: str) -> Optional[str]:
    """Returns:
    Model response or None if failed."""
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
                    "top_k": 10,
                },
            },
            timeout=10,
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"[ERROR] Ollama HTTP {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        print("[ERROR] Ollama not running. Start with: ollama serve")
        return None
    except Exception as e:
        print(f"[ERROR] Ollama query: {e}")
        return None


def format_response(response: str) -> str:
    """
    Clean and format LLM response.

    Args:
        response: Raw model response.

    Returns:
        Formatted response with alerts if needed.
    """
    if not response:
        return "[ALERTA] No response from model"
    # Trim to max length
    if len(response) > CONFIG["max_response_length"]:
        response = response[: CONFIG["max_response_length"]] + "..."
    # Add alert for uncertain responses
    uncertain_keywords = [
        "no estoy seguro",
        "quizás",
        "tal vez",
        "podría ser",
        "maybe",
        "perhaps",
    ]
    if any(kw in response.lower() for kw in uncertain_keywords):
        if not response.startswith("[ALERTA"):
            response = f"[ALERTA] {response}"
    return response


# ============================================================================
# ORCHESTRATION FUNCTION (The main workflow)
# ============================================================================
def process_screen() -> Tuple[str, str, float]:
    """
    Complete pipeline: capture -> OCR -> analyze -> respond.

    Returns:
        Tuple of (question_text, response_text, confidence_score).
    """
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


# UTILITY FUNCTIONS
# ============================================================================
def manual_capture_and_respond() -> None:
    """Manual trigger for testing."""
    print("\n📸 Capturing screen...")
    question, response, confidence = process_screen()

    print(f"\n{'=' * 50}")
    print(f"📝 DETECTED (conf: {confidence:.1f}%):")
    print(f"{'=' * 50}")
    print(question[:200] if question else "(none)")

    print(f"\n{'=' * 50}")
    print(f"💡 RESPONSE:")
    print(f"{'=' * 50}")
    print(response)
    print(f"{'=' * 50}\n")


def check_ollama() -> bool:
    """Verify Ollama is running and model is available."""
    try:
        response = requests.get(f"{CONFIG['ollama_host']}/api/tags", timeout=2)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            if CONFIG["ollama_model"] in models:
                print(f"✅ Ollama ready (model: {CONFIG['ollama_model']})")
                return True
            else:
                print(f"⚠️  Model '{CONFIG['ollama_model']}' not found. Pull it:")
                print(f"   ollama pull {CONFIG['ollama_model']}")
                return False
        return False
    except:
        print("⚠️  Ollama not running. Start with: ollama serve")
        return False


# MAIN LOOP (Simple and functional)
# ============================================================================
def run_assistant():
    """
    Main assistant loop with change detection.
    Uses functional composition and minimal state.
    """
    print("\n" + "=" * 50)
    print("IQ-Edge Interview Assistant")
    print("=" * 50)
    if not check_ollama():
        print("\n⚠️  Continuing but responses will fail...")
    print("\n⚙️  Configuration:")
    print(f"   - Screenshot every {CONFIG['screenshot_interval']}s")
    print(f"   - OCR languages: {CONFIG['ocr_languages']}")
    print(f"   - Model: {CONFIG['ollama_model']}")
    print("\n⌨️  Controls:")
    print("   - Press Ctrl+C to stop")
    print("   - Run manual_capture_and_respond() for single capture")
    print("\n" + "=" * 50)
    try:
        while True:
            # Only process if screen changed
            current_image = take_screenshot()
            if current_image and has_screen_changed(current_image):
                print("\n🔄 Screen changed, processing...")
                question, response, confidence = process_screen()
                if question:
                    print(f"\n📝 Q ({confidence:.0f}%): {question[:100]}...")
                    print(f"💡 A: {response}")
                else:
                    print("⏸️  No text detected")
            time.sleep(CONFIG["screenshot_interval"])
    except KeyboardInterrupt:
        print("\n\n👋 Assistant stopped")


# CLI INTERFACE
# ============================================================================
if __name__ == "__main__":
    """
    Usage examples:
    
    1. Run continuous assistant:
        python iq_edge.py
    
    2. Manual capture (in Python shell):
        >>> from iq_edge import manual_capture_and_respond
        >>> manual_capture_and_respond()
    
    3. One-off processing:
        >>> from iq_edge import process_screen
        >>> question, answer, conf = process_screen()
        >>> print(answer)
    """
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        manual_capture_and_respond()
    else:
        run_assistant()
