"""
Integration tests for ShadowView.
Run with: uv run python test_integration.py
"""

import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image

import main


class TestPreprocessing(unittest.TestCase):
    """Tests for image preprocessing."""

    def test_preprocess_returns_gray_image(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = main.preprocess_image(image)
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.dtype, np.uint8)

    def test_preprocess_upscales_2x(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = main.preprocess_image(image)
        self.assertEqual(result.shape, (200, 200))


class TestTextExtraction(unittest.TestCase):
    """Tests for OCR text extraction."""

    def test_extract_returns_tuple(self):
        image = Image.new("RGB", (100, 100), color="white")
        result = main.extract_text_from_image(image)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], float)

    def test_extract_empty_image_returns_zero_confidence(self):
        image = Image.new("RGB", (100, 100), color="white")
        text, conf = main.extract_text_from_image(image)
        self.assertEqual(conf, 0.0)


class TestContentAnalysis(unittest.TestCase):
    """Tests for content type detection."""

    def test_detects_iq_pattern(self):
        text = "pattern sequence next complete matrix"
        self.assertEqual(main.analyze_content(text), "iq_pattern")

    def test_detects_math(self):
        text = "calculate number equation solve math problem"
        self.assertEqual(main.analyze_content(text), "iq_math")

    def test_detects_code(self):
        text = "def function import class code"
        self.assertEqual(main.analyze_content(text), "code")

    def test_general_for_unknown(self):
        text = "hello world testing simple items today"
        self.assertEqual(main.analyze_content(text), "general")


class TestPromptBuilding(unittest.TestCase):
    """Tests for LLM prompt generation."""

    def test_prompt_contains_content(self):
        text = "What is 2+2?"
        prompt = main.build_dynamic_prompt(text, "iq_math")
        self.assertIn(text, prompt)
        self.assertIn("Answer (max 2 lines)", prompt)

    def test_prompt_respects_max_length(self):
        long_text = "word " * 200
        prompt = main.build_dynamic_prompt(long_text, "general")
        self.assertLess(len(prompt), 2000)


class TestResponseFormatting(unittest.TestCase):
    """Tests for response formatting."""

    def test_truncates_long_response(self):
        long_response = "a" * 500
        result = main.format_response(long_response)
        self.assertLessEqual(len(result), main.CONFIG["max_response_length"] + 3)

    def test_adds_alerta_for_uncertain(self):
        uncertain_phrases = ["not sure", "maybe", "perhaps"]
        for phrase in uncertain_phrases:
            result = main.format_response(f"The answer is {phrase}")
            self.assertTrue(result.startswith("[ALERTA]"))

    def test_empty_response(self):
        result = main.format_response("")
        self.assertEqual(result, "[ALERTA] No response")


class TestCaptureSaving(unittest.TestCase):
    """Tests for capture saving functionality."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.original_dir = main._captures_dir
        main._captures_dir = self.test_dir

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        main._captures_dir = self.original_dir

    def test_save_creates_directory(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.assertFalse(self.test_dir.exists())
        image = Image.new("RGB", (100, 100))
        main.save_capture(image)
        self.assertTrue(self.test_dir.exists())

    def test_save_returns_path(self):
        image = Image.new("RGB", (100, 100))
        path = main.save_capture(image)
        self.assertIsInstance(path, Path)
        self.assertEqual(path.suffix, ".png")

    def test_save_filename_format(self):
        image = Image.new("RGB", (100, 100))
        path = main.save_capture(image)
        self.assertTrue(path.name.startswith("capture_"))
        self.assertTrue(path.name.endswith(".png"))

    def test_save_creates_valid_image(self):
        image = Image.new("RGB", (50, 50), color="red")
        path = main.save_capture(image)
        loaded = Image.open(path)
        self.assertEqual(loaded.size, (50, 50))


class TestScreenChangeDetection(unittest.TestCase):
    """Tests for screen change detection."""

    def setUp(self):
        self.original_path = main._last_capture_path
        self.original_hash = main._last_screenshot_hash
        main._last_capture_path = None
        main._last_screenshot_hash = None

    def tearDown(self):
        main._last_capture_path = self.original_path
        main._last_screenshot_hash = self.original_hash

    def test_detects_change_on_first_image(self):
        image = Image.new("RGB", (100, 100))
        self.assertTrue(main.has_screen_changed(image))

    def test_detects_no_change_for_same_image(self):
        image = Image.new("RGB", (100, 100))
        main._last_capture_path = main.save_capture(image)
        main._last_screenshot_hash = hashlib.md5(image.tobytes()).hexdigest()
        self.assertFalse(main.has_screen_changed(image))

    def test_detects_change_for_different_image(self):
        image1 = Image.new("RGB", (100, 100), color="red")
        image2 = Image.new("RGB", (100, 100), color="blue")
        main._last_capture_path = main.save_capture(image1)
        main._last_screenshot_hash = hashlib.md5(image1.tobytes()).hexdigest()
        self.assertTrue(main.has_screen_changed(image2))


class TestRectangleOperations(unittest.TestCase):
    """Tests for rectangle helper functions."""

    def test_normalize_rectangle(self):
        left, top, right, bottom = main.normalize_rectangle(100, 200, 50, 150)
        self.assertEqual(left, 50)
        self.assertEqual(right, 100)
        self.assertEqual(top, 150)
        self.assertEqual(bottom, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
