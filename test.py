"""
IQ-Edge: Test Suite
Testing the screenshot capture and pipeline processing.
"""

from main import (
    take_screenshot,
    extract_text_from_image,
    analyze_content,
    build_dynamic_prompt,
    query_ollama,
    format_response,
)
import json
import os
from datetime import datetime


def save_pipeline_phases():
    """
    Capture screenshot and save all pipeline phases to individual files.
    Creates timestamped output directory with phase files.
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"pipeline_phases_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("🧪 Testing take_screenshot() and pipeline")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)

    # Phase 1: Take screenshot
    print("\n[FASE 1] Capturando pantalla...")
    image = take_screenshot()

    if not image:
        print("[ERROR] Screenshot failed")
        return

    # Save the screenshot
    screenshot_path = os.path.join(output_dir, "01_screenshot.png")
    image.save(screenshot_path)
    print(f"✅ Screenshot guardado: {screenshot_path}")
    print(f"   Dimensiones: {image.size}")

    # Phase 2: Extract text (OCR)
    print("\n[FASE 2] Extrayendo texto con OCR...")
    text, confidence = extract_text_from_image(image)

    phase_2_file = os.path.join(output_dir, "02_ocr_result.txt")
    with open(phase_2_file, "w", encoding="utf-8") as f:
        f.write(f"CONFIDENCE: {confidence:.1f}%\n")
        f.write(f"{'=' * 60}\n")
        f.write(text if text else "[INFO] No text detected in screenshot")

    if text:
        print(f"✅ Texto extraído (confianza: {confidence:.1f}%):")
        print(f"   {text[:150]}...")
        print(f"   Guardado en: {phase_2_file}")
    else:
        print("[INFO] No text detected in screenshot")
        print(f"   Guardado en: {phase_2_file}")

    # Phase 3: Analyze content
    print("\n[FASE 3] Analizando contenido...")
    content_type = analyze_content(text) if text else "general"

    phase_3_file = os.path.join(output_dir, "03_content_analysis.txt")
    with open(phase_3_file, "w", encoding="utf-8") as f:
        f.write(f"Content Type: {content_type}\n")

    print(f"✅ Tipo de contenido detectado: {content_type}")
    print(f"   Guardado en: {phase_3_file}")

    # Phase 4: Build dynamic prompt
    print("\n[FASE 4] Construyendo prompt dinámico...")
    prompt = build_dynamic_prompt(text, content_type) if text else ""

    phase_4_file = os.path.join(output_dir, "04_dynamic_prompt.txt")
    with open(phase_4_file, "w", encoding="utf-8") as f:
        f.write(prompt if prompt else "[INFO] No prompt generated")

    if prompt:
        print(f"✅ Prompt construido ({len(prompt)} caracteres)")
        print(f"   Preview: {prompt[:100]}...")
        print(f"   Guardado en: {phase_4_file}")
    else:
        print("[INFO] No prompt generated")
        print(f"   Guardado en: {phase_4_file}")

    # Phase 5: Query Ollama
    print("\n[FASE 5] Consultando modelo Ollama...")
    raw_response = query_ollama(prompt) if prompt else None

    phase_5_file = os.path.join(output_dir, "05_ollama_raw_response.txt")
    with open(phase_5_file, "w", encoding="utf-8") as f:
        f.write(raw_response if raw_response else "[INFO] No response from Ollama")

    if raw_response:
        print("✅ Respuesta cruda recibida:")
        print(f"   {raw_response}")
        print(f"   Guardado en: {phase_5_file}")
    else:
        print("[INFO] No response from Ollama")
        print(f"   Guardado en: {phase_5_file}")

    # Phase 6: Format response
    print("\n[FASE 6] Formateando respuesta...")
    formatted_response = (
        format_response(raw_response) if raw_response else "[ERROR] No response"
    )

    phase_6_file = os.path.join(output_dir, "06_formatted_response.txt")
    with open(phase_6_file, "w", encoding="utf-8") as f:
        f.write(formatted_response)

    print("✅ Respuesta formateada:")
    print(f"   {formatted_response}")
    print(f"   Guardado en: {phase_6_file}")

    # Save comprehensive JSON summary
    print("\n[GUARDANDO RESUMEN JSON]...")
    summary = {
        "timestamp": timestamp,
        "output_directory": output_dir,
        "phases": {
            "1_screenshot": {
                "file": "01_screenshot.png",
                "dimensions": image.size,
            },
            "2_ocr": {
                "file": "02_ocr_result.txt",
                "text_length": len(text) if text else 0,
                "confidence": confidence,
            },
            "3_content_analysis": {
                "file": "03_content_analysis.txt",
                "content_type": content_type,
            },
            "4_dynamic_prompt": {
                "file": "04_dynamic_prompt.txt",
                "prompt_length": len(prompt),
            },
            "5_ollama_raw": {
                "file": "05_ollama_raw_response.txt",
                "response_length": len(raw_response) if raw_response else 0,
            },
            "6_formatted_response": {
                "file": "06_formatted_response.txt",
                "response_length": len(formatted_response),
            },
        },
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Resumen guardado: {summary_path}")

    print("\n" + "=" * 60)
    print("✅ Test completado exitosamente")
    print(f"📁 Archivos guardados en: {os.path.abspath(output_dir)}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    save_pipeline_phases()
