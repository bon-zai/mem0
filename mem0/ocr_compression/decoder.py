"""
OCR Decoder - Converts image tokens back to text

This module handles the decoding of image representations back to text content.
It supports multiple OCR backends for flexibility and accuracy.
"""

import io
import logging
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)


class OCRDecoder:
    """
    Decodes image tokens back to text content.

    Supports multiple OCR backends:
    - EasyOCR (default, deep learning-based)
    - PaddleOCR (fast and accurate)
    - Tesseract (traditional, fallback)
    - Janus-Pro (experimental, if available)
    """

    def __init__(
        self,
        backend: str = "easyocr",
        languages: list = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the OCR decoder.

        Args:
            backend: OCR backend to use ('easyocr', 'paddleocr', 'tesseract', 'janus')
            languages: List of language codes (default: ['en'])
            model_path: Path to custom OCR model (for Janus backend)
        """
        self.backend = backend.lower()
        self.languages = languages or ['en']
        self.model_path = model_path
        self.reader = None

        # Initialize the OCR engine
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the selected OCR backend."""
        try:
            if self.backend == "easyocr":
                self._initialize_easyocr()
            elif self.backend == "paddleocr":
                self._initialize_paddleocr()
            elif self.backend == "tesseract":
                self._initialize_tesseract()
            elif self.backend == "janus":
                self._initialize_janus()
            else:
                logger.warning(f"Unknown backend '{self.backend}', falling back to EasyOCR")
                self.backend = "easyocr"
                self._initialize_easyocr()
        except Exception as e:
            logger.warning(f"Failed to initialize {self.backend}: {e}, trying fallback")
            # Fallback to simpler backend
            self._initialize_fallback()

    def _initialize_easyocr(self):
        """Initialize EasyOCR backend."""
        try:
            import easyocr
            self.reader = easyocr.Reader(self.languages, gpu=False)
            logger.info(f"Initialized EasyOCR with languages: {self.languages}")
        except ImportError:
            logger.warning("EasyOCR not installed. Install with: pip install easyocr")
            raise

    def _initialize_paddleocr(self):
        """Initialize PaddleOCR backend."""
        try:
            from paddleocr import PaddleOCR
            self.reader = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False,
                use_gpu=False
            )
            logger.info("Initialized PaddleOCR")
        except ImportError:
            logger.warning("PaddleOCR not installed. Install with: pip install paddleocr")
            raise

    def _initialize_tesseract(self):
        """Initialize Tesseract backend."""
        try:
            import pytesseract
            # Test if tesseract is available
            pytesseract.get_tesseract_version()
            self.reader = pytesseract
            logger.info("Initialized Tesseract OCR")
        except (ImportError, Exception) as e:
            logger.warning(f"Tesseract not available: {e}")
            raise

    def _initialize_janus(self):
        """Initialize Janus-Pro OCR backend (experimental)."""
        try:
            # This is a placeholder for Janus-Pro integration
            # In production, this would load the Janus-Pro-7B model
            logger.warning("Janus-Pro backend is experimental and not fully implemented")
            raise NotImplementedError("Janus-Pro backend coming soon")
        except Exception as e:
            logger.error(f"Failed to initialize Janus-Pro: {e}")
            raise

    def _initialize_fallback(self):
        """Initialize fallback OCR backend."""
        # Try backends in order of preference
        for backend_name in ["easyocr", "paddleocr", "tesseract"]:
            try:
                self.backend = backend_name
                if backend_name == "easyocr":
                    self._initialize_easyocr()
                elif backend_name == "paddleocr":
                    self._initialize_paddleocr()
                elif backend_name == "tesseract":
                    self._initialize_tesseract()
                logger.info(f"Fallback: using {backend_name}")
                return
            except Exception:
                continue

        # If all fail, use a dummy decoder
        logger.error("No OCR backend available! Using dummy decoder.")
        self.backend = "dummy"
        self.reader = None

    def decode(self, image_bytes: bytes) -> str:
        """
        Decode image bytes back to text.

        Args:
            image_bytes: Image data as bytes

        Returns:
            Decoded text content

        Raises:
            ValueError: If image_bytes is empty or invalid
        """
        if not image_bytes:
            raise ValueError("Cannot decode empty image")

        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))

        # Perform OCR based on backend
        text = self._perform_ocr(image)

        logger.debug(f"Decoded {len(image_bytes)} bytes to {len(text)} chars using {self.backend}")

        return text.strip()

    def _perform_ocr(self, image: Image.Image) -> str:
        """
        Perform OCR on image using selected backend.

        Args:
            image: PIL Image to process

        Returns:
            Extracted text
        """
        if self.backend == "easyocr":
            return self._easyocr_extract(image)
        elif self.backend == "paddleocr":
            return self._paddleocr_extract(image)
        elif self.backend == "tesseract":
            return self._tesseract_extract(image)
        elif self.backend == "janus":
            return self._janus_extract(image)
        elif self.backend == "dummy":
            return self._dummy_extract(image)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _easyocr_extract(self, image: Image.Image) -> str:
        """Extract text using EasyOCR."""
        import numpy as np

        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Perform OCR
        results = self.reader.readtext(img_array)

        # Combine all detected text
        text = ' '.join([result[1] for result in results])
        return text

    def _paddleocr_extract(self, image: Image.Image) -> str:
        """Extract text using PaddleOCR."""
        import numpy as np

        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Perform OCR
        results = self.reader.ocr(img_array, cls=True)

        # Extract text from results
        if results and results[0]:
            text = ' '.join([line[1][0] for line in results[0]])
            return text
        return ""

    def _tesseract_extract(self, image: Image.Image) -> str:
        """Extract text using Tesseract."""
        # Tesseract works directly with PIL Images
        text = self.reader.image_to_string(image)
        return text

    def _janus_extract(self, image: Image.Image) -> str:
        """Extract text using Janus-Pro (placeholder)."""
        # This would use the Janus-Pro-7B model for image understanding
        raise NotImplementedError("Janus-Pro extraction not yet implemented")

    def _dummy_extract(self, image: Image.Image) -> str:
        """Dummy extractor (returns placeholder)."""
        logger.warning("Using dummy OCR - no actual text extraction")
        return "<OCR Not Available>"

    def verify_accuracy(self, original_text: str, decoded_text: str) -> dict:
        """
        Verify decoding accuracy by comparing original and decoded text.

        Args:
            original_text: Original text before encoding
            decoded_text: Text after encode/decode cycle

        Returns:
            Dictionary with accuracy metrics
        """
        # Character-level accuracy
        original_chars = len(original_text)
        decoded_chars = len(decoded_text)

        # Simple character match (case-insensitive, whitespace normalized)
        orig_normalized = ' '.join(original_text.lower().split())
        dec_normalized = ' '.join(decoded_text.lower().split())

        # Calculate similarity using a simple method
        matches = sum(1 for a, b in zip(orig_normalized, dec_normalized) if a == b)
        max_len = max(len(orig_normalized), len(dec_normalized))

        accuracy = matches / max_len if max_len > 0 else 0

        return {
            "original_length": original_chars,
            "decoded_length": decoded_chars,
            "character_accuracy": accuracy,
            "accuracy_percentage": accuracy * 100,
            "is_acceptable": accuracy >= 0.95,  # 95% threshold
            "backend": self.backend,
        }
