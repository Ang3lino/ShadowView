#!/usr/bin/env python3
"""Test mínimo para identificar el error."""

import sys
import platform

print("=" * 50)
print("Test de Diagnóstico - ShadowView")
print("=" * 50)

print(f"\n🐍 Python: {sys.version}")
print(f"💻 Sistema: {platform.system()} {platform.machine()}")

# 1. Test básico de imports
print("\n1. Probando imports...")
try:
    import numpy

    print(f"   ✅ NumPy: {numpy.__version__}")
except Exception as e:
    print(f"   ❌ NumPy: {e}")

try:
    import PIL

    print(f"   ✅ Pillow: {PIL.__version__}")
except Exception as e:
    print(f"   ❌ Pillow: {e}")

try:
    import cv2

    print(f"   ✅ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"   ❌ OpenCV: {e}")

try:
    import pytesseract

    print(f"   ✅ Tesseract")
except Exception as e:
    print(f"   ❌ Tesseract: {e}")

# 2. Test de captura de pantalla
print("\n2. Probando captura de pantalla...")
try:
    import pyautogui

    print(f"   ✅ PyAutoGUI")

    # Intentar captura
    screenshot = pyautogui.screenshot()
    print(f"   ✅ Captura exitosa: {screenshot.size}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n✅ Test completado")
