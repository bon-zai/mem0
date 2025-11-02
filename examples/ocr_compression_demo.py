#!/usr/bin/env python3
"""
OCR Compression Demo

This script demonstrates the OCR-compressed memory system for Mem0.
It shows how to encode text to images, store them, and decode them back.

Usage:
    python examples/ocr_compression_demo.py
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_compression():
    """Demonstrate basic OCR compression."""
    from mem0.ocr_compression.encoder import OCREncoder
    from mem0.ocr_compression.decoder import OCRDecoder
    from mem0.ocr_compression.storage import ImageStorage

    print("\n" + "=" * 70)
    print("DEMO 1: Basic OCR Compression")
    print("=" * 70)

    # Sample memories
    memories = [
        "I love science fiction movies, especially Dune and Blade Runner.",
        "My favorite programming language is Python for data science.",
        "I prefer morning workouts over evening sessions.",
        "The best coffee shop in town is Blue Bottle Coffee.",
        "I'm learning machine learning and deep learning this year.",
    ]

    # Initialize components
    encoder = OCREncoder()
    storage = ImageStorage(storage_dir="/tmp/mem0_demo")

    print("\nEncoding and storing memories...")
    print("-" * 70)

    compression_stats = []

    for i, memory_text in enumerate(memories, 1):
        # Encode to image
        image_data = encoder.encode(memory_text)

        # Save to storage
        image_id = storage.save_image(
            image_data=image_data,
            user_id="demo_user",
            timestamp=f"2025-11-02T10:{i:02d}:00Z",
            memory_id=f"memory_{i}"
        )

        # Get compression stats
        stats = encoder.get_compression_stats(memory_text, image_data)

        compression_stats.append(stats)

        print(f"\nMemory {i}: '{memory_text[:50]}...'")
        print(f"  Text: {stats['text_bytes']} bytes")
        print(f"  Image: {stats['image_bytes']} bytes")
        print(f"  Compression: {stats['byte_compression_ratio']:.2f}x")
        print(f"  Saved as: {image_id}")

    # Calculate average compression
    avg_compression = sum(s['byte_compression_ratio'] for s in compression_stats) / len(compression_stats)
    total_text = sum(s['text_bytes'] for s in compression_stats)
    total_image = sum(s['image_bytes'] for s in compression_stats)

    print("\n" + "-" * 70)
    print(f"Summary:")
    print(f"  Total memories: {len(memories)}")
    print(f"  Total text bytes: {total_text}")
    print(f"  Total image bytes: {total_image}")
    print(f"  Average compression: {avg_compression:.2f}x")
    print(f"  Storage saved: {total_text - total_image} bytes")

    # Get storage stats
    storage_stats = storage.get_storage_stats()
    print(f"\nStorage location: {storage_stats['storage_dir']}")
    print(f"Images stored: {storage_stats['total_images']}")


def demo_encode_decode_accuracy():
    """Demonstrate encode/decode accuracy."""
    from mem0.ocr_compression.encoder import OCREncoder
    from mem0.ocr_compression.decoder import OCRDecoder

    print("\n" + "=" * 70)
    print("DEMO 2: Encode/Decode Accuracy Test")
    print("=" * 70)

    # Initialize
    encoder = OCREncoder()

    # Try to initialize decoder (may fail if no OCR backend)
    try:
        decoder = OCRDecoder(backend="easyocr")
        print("\nUsing EasyOCR backend (this may take a moment to load...)")
    except Exception as e:
        print(f"\nWarning: Could not load EasyOCR: {e}")
        print("Note: For full OCR functionality, install: pip install easyocr")
        print("\nSkipping decode accuracy test.")
        return

    # Test cases
    test_cases = [
        "Simple text: Hello, World!",
        "With numbers: User ID 12345, Score: 98.5%",
        "Technical: Machine learning models use neural networks.",
    ]

    print("\nTesting encode/decode accuracy...")
    print("-" * 70)

    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{text}'")

        # Encode
        image_data = encoder.encode(text)
        print(f"  Encoded to {len(image_data)} bytes")

        # Decode
        try:
            decoded = decoder.decode(image_data)
            print(f"  Decoded: '{decoded}'")

            # Verify accuracy
            accuracy_result = decoder.verify_accuracy(text, decoded)
            accuracy_pct = accuracy_result['accuracy_percentage']

            print(f"  Accuracy: {accuracy_pct:.1f}%")

            if accuracy_pct >= 95:
                print(f"  ✓ High accuracy!")
            elif accuracy_pct >= 90:
                print(f"  ⚠ Acceptable accuracy")
            else:
                print(f"  ✗ Low accuracy")

        except Exception as e:
            print(f"  ✗ Decode failed: {e}")


def demo_integration_example():
    """Show integration example with Mem0."""
    print("\n" + "=" * 70)
    print("DEMO 3: Integration Example (Conceptual)")
    print("=" * 70)

    example_code = '''
# Example: Using OCR compression with Mem0

from mem0.memory.main import Memory
from mem0.ocr_compression.memory_writer import MemoryWriter
from mem0.ocr_compression.memory_reader import MemoryReader

# Initialize Mem0
mem = Memory()

# Create OCR-enabled writer
writer = MemoryWriter(
    vector_store=mem.vector_store,
    embedding_model=mem.embedding_model,
    enable_ocr_compression=True
)

# Write a memory with OCR compression
result = writer.write_memory(
    content="I love Python programming and machine learning.",
    user_id="user123"
)

print(f"Compression ratio: {result['compression_stats']['byte_compression_ratio']:.2f}x")
print(f"Token savings: {result['compression_stats']['estimated_token_savings'] * 100:.1f}%")

# Create OCR-enabled reader
reader = MemoryReader(
    vector_store=mem.vector_store,
    embedding_model=mem.embedding_model,
    ocr_backend="easyocr"
)

# Query memories (will auto-decode OCR-compressed memories)
results = reader.query_memory(
    query="programming languages",
    user_id="user123",
    top_k=5
)

for memory in results['results']:
    print(f"Memory: {memory['memory']}")
    print(f"Score: {memory['score']:.3f}")
    print(f"OCR Compressed: {memory['ocr_compressed']}")
'''

    print("\nIntegration code example:")
    print("-" * 70)
    print(example_code)
    print("-" * 70)
    print("\nNote: This is a conceptual example showing how to integrate")
    print("the OCR compression system with Mem0's existing infrastructure.")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("OCR-Compressed Memory System - Demonstration")
    print("=" * 70)
    print("\nThis demo shows the OCR compression capabilities for Mem0.")
    print("It demonstrates text-to-image encoding, storage, and decoding.")

    try:
        # Demo 1: Basic compression
        demo_basic_compression()

        # Demo 2: Accuracy test (requires OCR backend)
        demo_encode_decode_accuracy()

        # Demo 3: Integration example
        demo_integration_example()

        print("\n" + "=" * 70)
        print("Demo completed successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Install OCR backend: pip install easyocr")
        print("2. Read the documentation: mem0/ocr_compression/README.md")
        print("3. Run tests: python -m mem0.ocr_compression.test_memory_system")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
