#!/usr/bin/env python3
"""
Basic validation test for OCR compression system.
Tests encoder functionality without requiring OCR backend.
"""

import sys
import logging
from pathlib import Path

# Add mem0 to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_encoder():
    """Test the OCR encoder."""
    logger.info("Testing OCR Encoder...")

    try:
        from mem0.ocr_compression.encoder import OCREncoder

        encoder = OCREncoder()
        logger.info("✓ OCREncoder imported successfully")

        # Test encoding
        test_text = "Hello, this is a test message for OCR compression."
        logger.info(f"Encoding text: '{test_text}'")

        image_bytes = encoder.encode(test_text)
        logger.info(f"✓ Encoded to {len(image_bytes)} bytes")

        # Get compression stats
        stats = encoder.get_compression_stats(test_text, image_bytes)
        logger.info(f"✓ Compression ratio: {stats['byte_compression_ratio']:.2f}x")
        logger.info(f"✓ Text bytes: {stats['text_bytes']}, Image bytes: {stats['image_bytes']}")

        return True

    except Exception as e:
        logger.error(f"✗ Encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_storage():
    """Test image storage."""
    logger.info("\nTesting Image Storage...")

    try:
        from mem0.ocr_compression.storage import ImageStorage
        from mem0.ocr_compression.encoder import OCREncoder
        import tempfile
        import os

        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="mem0_test_")
        logger.info(f"Using temp directory: {temp_dir}")

        storage = ImageStorage(storage_dir=temp_dir)
        logger.info("✓ ImageStorage initialized")

        # Create test image
        encoder = OCREncoder()
        test_text = "Test storage functionality"
        image_bytes = encoder.encode(test_text)

        # Save image
        image_id = storage.save_image(
            image_data=image_bytes,
            user_id="test_user",
            timestamp="2025-11-02T10:00:00Z",
            memory_id="test_123"
        )
        logger.info(f"✓ Saved image: {image_id}")

        # Load image
        loaded_bytes = storage.load_image(image_id)
        logger.info(f"✓ Loaded image: {len(loaded_bytes)} bytes")

        # Verify
        if loaded_bytes == image_bytes:
            logger.info("✓ Image data verified (matches original)")
        else:
            logger.warning("⚠ Image data mismatch")

        # Get stats
        stats = storage.get_storage_stats()
        logger.info(f"✓ Storage stats: {stats['total_images']} images, {stats['total_size_bytes']} bytes")

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        logger.info(f"✓ Cleaned up temp directory")

        return True

    except Exception as e:
        logger.error(f"✗ Storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test that all modules can be imported."""
    logger.info("\nTesting Module Imports...")

    modules = [
        "mem0.ocr_compression",
        "mem0.ocr_compression.encoder",
        "mem0.ocr_compression.decoder",
        "mem0.ocr_compression.storage",
        "mem0.ocr_compression.memory_writer",
        "mem0.ocr_compression.memory_reader",
        "mem0.ocr_compression.migration_script",
    ]

    success = True
    for module_name in modules:
        try:
            __import__(module_name)
            logger.info(f"✓ {module_name}")
        except Exception as e:
            logger.error(f"✗ {module_name}: {e}")
            success = False

    return success


def main():
    """Run all basic tests."""
    print("=" * 70)
    print("OCR Compression System - Basic Validation Tests")
    print("=" * 70)

    results = []

    # Test imports
    results.append(("Module Imports", test_imports()))

    # Test encoder
    results.append(("Encoder", test_encoder()))

    # Test storage
    results.append(("Storage", test_storage()))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:20s} : {status}")

    print("=" * 70)
    print(f"Result: {passed}/{total} tests passed")

    if passed == total:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
