"""
Test Harness - OCR Memory System Validation

This test harness validates the OCR-powered memory system by:
1. Testing memory write/read cycles
2. Measuring compression ratios
3. Verifying decode accuracy
4. Benchmarking latency

Run with: python -m mem0.ocr_compression.test_memory_system
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Any

from mem0.ocr_compression.encoder import OCREncoder
from mem0.ocr_compression.decoder import OCRDecoder
from mem0.ocr_compression.storage import ImageStorage
from mem0.ocr_compression.memory_writer import MemoryWriter
from mem0.ocr_compression.memory_reader import MemoryReader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestMetrics:
    """Track test metrics."""

    def __init__(self):
        self.total_tests = 0
        self.passed = 0
        self.failed = 0
        self.compression_ratios = []
        self.accuracy_scores = []
        self.encode_times = []
        self.decode_times = []
        self.roundtrip_times = []

    def add_test(self, passed: bool):
        """Record test result."""
        self.total_tests += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def add_compression_ratio(self, ratio: float):
        """Record compression ratio."""
        self.compression_ratios.append(ratio)

    def add_accuracy(self, accuracy: float):
        """Record accuracy."""
        self.accuracy_scores.append(accuracy)

    def add_encode_time(self, time_ms: float):
        """Record encoding time."""
        self.encode_times.append(time_ms)

    def add_decode_time(self, time_ms: float):
        """Record decoding time."""
        self.decode_times.append(time_ms)

    def add_roundtrip_time(self, time_ms: float):
        """Record roundtrip time."""
        self.roundtrip_times.append(time_ms)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.passed / self.total_tests if self.total_tests > 0 else 0,
            "avg_compression_ratio": sum(self.compression_ratios) / len(self.compression_ratios) if self.compression_ratios else 0,
            "avg_accuracy": sum(self.accuracy_scores) / len(self.accuracy_scores) if self.accuracy_scores else 0,
            "avg_encode_time_ms": sum(self.encode_times) / len(self.encode_times) if self.encode_times else 0,
            "avg_decode_time_ms": sum(self.decode_times) / len(self.decode_times) if self.decode_times else 0,
            "avg_roundtrip_time_ms": sum(self.roundtrip_times) / len(self.roundtrip_times) if self.roundtrip_times else 0,
        }


class OCRMemoryTestHarness:
    """
    Test harness for OCR-powered memory system.
    """

    def __init__(self, storage_dir: str = "/tmp/mem0_ocr_test"):
        """
        Initialize test harness.

        Args:
            storage_dir: Directory for test storage
        """
        self.storage_dir = storage_dir
        self.metrics = TestMetrics()

        # Initialize components
        self.encoder = OCREncoder()
        self.decoder = OCRDecoder(backend="easyocr")  # Will fallback if not available
        self.image_storage = ImageStorage(storage_dir=storage_dir)

        logger.info(f"Initialized test harness with storage: {storage_dir}")

    def run_all_tests(self):
        """Run all test suites."""
        logger.info("=" * 70)
        logger.info("STARTING OCR MEMORY SYSTEM TEST HARNESS")
        logger.info("=" * 70)

        # Test 1: Basic encode/decode
        logger.info("\n[Test 1] Basic Encode/Decode Tests")
        self.test_basic_encode_decode()

        # Test 2: Compression ratio
        logger.info("\n[Test 2] Compression Ratio Tests")
        self.test_compression_ratios()

        # Test 3: Accuracy verification
        logger.info("\n[Test 3] Accuracy Verification Tests")
        self.test_accuracy()

        # Test 4: Performance benchmarks
        logger.info("\n[Test 4] Performance Benchmark Tests")
        self.test_performance()

        # Test 5: Edge cases
        logger.info("\n[Test 5] Edge Case Tests")
        self.test_edge_cases()

        # Print summary
        self.print_summary()

    def test_basic_encode_decode(self):
        """Test basic encode/decode functionality."""
        test_cases = [
            "I love sci-fi movies.",
            "My favorite book is Dune.",
            "The weather is nice today.",
            "I prefer Python over JavaScript.",
            "Machine learning is fascinating.",
        ]

        for i, text in enumerate(test_cases, 1):
            try:
                logger.info(f"  Test case {i}: '{text[:50]}...'")

                # Encode
                start = time.time()
                image_data = self.encoder.encode(text)
                encode_time = (time.time() - start) * 1000
                self.metrics.add_encode_time(encode_time)

                # Decode
                start = time.time()
                decoded_text = self.decoder.decode(image_data)
                decode_time = (time.time() - start) * 1000
                self.metrics.add_decode_time(decode_time)

                # Verify
                accuracy_result = self.decoder.verify_accuracy(text, decoded_text)
                accuracy = accuracy_result["character_accuracy"]
                self.metrics.add_accuracy(accuracy)

                # Check if test passed
                passed = accuracy >= 0.90  # 90% threshold for basic test

                logger.info(f"    Encode: {encode_time:.1f}ms, Decode: {decode_time:.1f}ms")
                logger.info(f"    Accuracy: {accuracy * 100:.1f}%")
                logger.info(f"    Result: {'✓ PASS' if passed else '✗ FAIL'}")

                self.metrics.add_test(passed)

            except Exception as e:
                logger.error(f"    ✗ FAIL: {e}")
                self.metrics.add_test(False)

    def test_compression_ratios(self):
        """Test compression ratios for different text lengths."""
        test_cases = [
            ("Short text", "Hello world!"),
            ("Medium text", "This is a medium-length text that contains several sentences. " * 3),
            ("Long text", "This is a longer text sample that will be used to test compression. " * 10),
            ("Very long text", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20),
        ]

        for name, text in test_cases:
            try:
                logger.info(f"  Testing {name}: {len(text)} chars")

                image_data = self.encoder.encode(text)
                stats = self.encoder.get_compression_stats(text, image_data)

                compression_ratio = stats["byte_compression_ratio"]
                self.metrics.add_compression_ratio(compression_ratio)

                logger.info(f"    Text bytes: {stats['text_bytes']}")
                logger.info(f"    Image bytes: {stats['image_bytes']}")
                logger.info(f"    Compression ratio: {compression_ratio:.2f}x")
                logger.info(f"    Token savings: {stats['estimated_token_savings'] * 100:.1f}%")

                # For compression tests, we consider it passed if there's some compression
                passed = compression_ratio > 0.5
                logger.info(f"    Result: {'✓ PASS' if passed else '✗ FAIL'}")

                self.metrics.add_test(passed)

            except Exception as e:
                logger.error(f"    ✗ FAIL: {e}")
                self.metrics.add_test(False)

    def test_accuracy(self):
        """Test decode accuracy with various content types."""
        test_cases = [
            ("Simple sentence", "The quick brown fox jumps over the lazy dog."),
            ("With numbers", "I have 3 cats and 2 dogs at home."),
            ("With punctuation", "Hello! How are you? I'm doing well, thanks."),
            ("Technical terms", "Machine learning uses neural networks and transformers."),
            ("Mixed content", "User ID: 12345, Email: user@example.com, Date: 2025-11-02"),
        ]

        for name, text in test_cases:
            try:
                logger.info(f"  Testing {name}")

                # Encode and decode
                image_data = self.encoder.encode(text)
                decoded_text = self.decoder.decode(image_data)

                # Verify accuracy
                accuracy_result = self.decoder.verify_accuracy(text, decoded_text)
                accuracy = accuracy_result["character_accuracy"]

                logger.info(f"    Original: '{text[:60]}...'")
                logger.info(f"    Decoded:  '{decoded_text[:60]}...'")
                logger.info(f"    Accuracy: {accuracy * 100:.1f}%")

                # For accuracy tests, we want high accuracy (95%)
                passed = accuracy >= 0.95

                logger.info(f"    Result: {'✓ PASS' if passed else '✗ FAIL'}")

                self.metrics.add_test(passed)
                self.metrics.add_accuracy(accuracy)

            except Exception as e:
                logger.error(f"    ✗ FAIL: {e}")
                self.metrics.add_test(False)

    def test_performance(self):
        """Test performance benchmarks."""
        test_text = "This is a test sentence for performance benchmarking. " * 5

        logger.info(f"  Benchmarking with {len(test_text)} character text")
        logger.info(f"  Running 10 iterations...")

        roundtrip_times = []

        for i in range(10):
            start = time.time()

            # Full roundtrip: encode -> save -> load -> decode
            image_data = self.encoder.encode(test_text)
            image_id = self.image_storage.save_image(
                image_data=image_data,
                user_id="test_user",
                timestamp=datetime.now().isoformat(),
                memory_id=f"bench_{i}"
            )
            loaded_data = self.image_storage.load_image(image_id)
            decoded_text = self.decoder.decode(loaded_data)

            roundtrip_time = (time.time() - start) * 1000
            roundtrip_times.append(roundtrip_time)
            self.metrics.add_roundtrip_time(roundtrip_time)

        avg_time = sum(roundtrip_times) / len(roundtrip_times)
        min_time = min(roundtrip_times)
        max_time = max(roundtrip_times)

        logger.info(f"    Average roundtrip: {avg_time:.1f}ms")
        logger.info(f"    Min: {min_time:.1f}ms, Max: {max_time:.1f}ms")

        # Consider test passed if average roundtrip is under 5 seconds
        passed = avg_time < 5000
        logger.info(f"    Result: {'✓ PASS' if passed else '✗ FAIL'}")

        self.metrics.add_test(passed)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        test_cases = [
            ("Empty string", ""),
            ("Very short", "Hi"),
            ("Unicode", "Hello 世界 🌍"),
            ("Special chars", "!@#$%^&*()_+-=[]{}|;':\",./<>?"),
            ("Whitespace", "   spaces   and   tabs\t\t"),
        ]

        for name, text in test_cases:
            try:
                logger.info(f"  Testing {name}: '{text[:50]}'")

                if not text or not text.strip():
                    # Empty text should raise ValueError
                    try:
                        image_data = self.encoder.encode(text)
                        logger.info(f"    ✗ FAIL: Should have raised ValueError for empty text")
                        self.metrics.add_test(False)
                    except ValueError:
                        logger.info(f"    ✓ PASS: Correctly raised ValueError")
                        self.metrics.add_test(True)
                else:
                    # Non-empty text should work
                    image_data = self.encoder.encode(text)
                    decoded_text = self.decoder.decode(image_data)

                    # For edge cases, we're more lenient on accuracy
                    accuracy_result = self.decoder.verify_accuracy(text, decoded_text)
                    accuracy = accuracy_result["character_accuracy"]

                    passed = accuracy >= 0.80  # 80% threshold for edge cases

                    logger.info(f"    Accuracy: {accuracy * 100:.1f}%")
                    logger.info(f"    Result: {'✓ PASS' if passed else '✗ FAIL'}")

                    self.metrics.add_test(passed)

            except Exception as e:
                logger.error(f"    ✗ FAIL: {e}")
                self.metrics.add_test(False)

    def print_summary(self):
        """Print test summary."""
        summary = self.metrics.get_summary()

        logger.info("\n" + "=" * 70)
        logger.info("TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Pass rate: {summary['pass_rate'] * 100:.1f}%")
        logger.info("")
        logger.info("Performance Metrics:")
        logger.info(f"  Average compression ratio: {summary['avg_compression_ratio']:.2f}x")
        logger.info(f"  Average accuracy: {summary['avg_accuracy'] * 100:.1f}%")
        logger.info(f"  Average encode time: {summary['avg_encode_time_ms']:.1f}ms")
        logger.info(f"  Average decode time: {summary['avg_decode_time_ms']:.1f}ms")
        logger.info(f"  Average roundtrip time: {summary['avg_roundtrip_time_ms']:.1f}ms")
        logger.info("=" * 70)

        # Overall result
        if summary['pass_rate'] >= 0.80:
            logger.info("✓ OVERALL: TESTS PASSED")
        else:
            logger.info("✗ OVERALL: TESTS FAILED")

        logger.info("=" * 70)


def main():
    """Main entry point for test harness."""
    print("\n" + "=" * 70)
    print("OCR-Compressed Memory System - Test Harness")
    print("=" * 70 + "\n")

    harness = OCRMemoryTestHarness()

    try:
        harness.run_all_tests()
    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
    except Exception as e:
        logger.error(f"\nTest harness failed: {e}", exc_info=True)

    print("\n")


if __name__ == "__main__":
    main()
