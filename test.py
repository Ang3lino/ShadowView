import platform
from typing import Optional

from PIL import Image

try:
    import pyautogui
except ImportError:
    pyautogui = None

try:
    from mss import mss
except ImportError:
    mss = None


#
#
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
    else:  # Linux/Windows
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


img = take_screenshot()

if img is None:
    print("[INFO] take_screenshot() returned None")
else:
    img.save("screenshot.png")
    print(f"✅ Screenshot saved to screenshot.png ({img.size})")
