"""
OCR Encoder - Converts text to compressed image tokens

This module handles the encoding of text content into optimized image representations.
The images are designed to be compact while maintaining OCR-readability for accurate
text reconstruction.
"""

import io
import logging
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class OCREncoder:
    """
    Encodes text content into compressed image tokens.

    This encoder creates optimized images from text, balancing compression ratio
    with OCR accuracy. Images are rendered with high-contrast, OCR-friendly fonts
    at resolutions optimized for both file size and readability.
    """

    def __init__(
        self,
        font_size: int = 12,
        max_width: int = 800,
        background_color: str = "white",
        text_color: str = "black",
        padding: int = 10,
        compression_quality: int = 85,
    ):
        """
        Initialize the OCR encoder.

        Args:
            font_size: Font size for text rendering (default: 12)
            max_width: Maximum image width in pixels (default: 800)
            background_color: Background color for images (default: "white")
            text_color: Text color (default: "black")
            padding: Padding around text in pixels (default: 10)
            compression_quality: JPEG compression quality 1-95 (default: 85)
        """
        self.font_size = font_size
        self.max_width = max_width
        self.background_color = background_color
        self.text_color = text_color
        self.padding = padding
        self.compression_quality = compression_quality

        # Try to load a monospace font for better OCR accuracy
        try:
            # Try common monospace fonts
            for font_name in [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                "/System/Library/Fonts/Courier.dfont",  # macOS
                "C:\\Windows\\Fonts\\consola.ttf",  # Windows
            ]:
                try:
                    self.font = ImageFont.truetype(font_name, self.font_size)
                    logger.info(f"Loaded font: {font_name}")
                    break
                except (OSError, IOError):
                    continue
            else:
                # Fallback to default font
                self.font = ImageFont.load_default()
                logger.warning("Using default font (may reduce OCR accuracy)")
        except Exception as e:
            logger.warning(f"Font loading error: {e}, using default font")
            self.font = ImageFont.load_default()

    def encode(self, text: str) -> bytes:
        """
        Encode text into a compressed image representation.

        Args:
            text: The text content to encode

        Returns:
            bytes: The encoded image as bytes (JPEG format)

        Raises:
            ValueError: If text is empty or None
        """
        if not text or not text.strip():
            raise ValueError("Cannot encode empty text")

        # Clean and prepare text
        text = text.strip()

        # Create image with text
        image = self._render_text_to_image(text)

        # Compress to bytes
        image_bytes = self._compress_image(image)

        logger.debug(
            f"Encoded {len(text)} chars to {len(image_bytes)} bytes "
            f"(compression ratio: {len(text) / len(image_bytes):.2f}x)"
        )

        return image_bytes

    def _render_text_to_image(self, text: str) -> Image.Image:
        """
        Render text to a PIL Image.

        Args:
            text: Text to render

        Returns:
            PIL Image containing the rendered text
        """
        # Split text into lines based on max width
        lines = self._wrap_text(text)

        # Calculate image dimensions
        line_height = self.font_size + 4  # Add line spacing
        image_height = (len(lines) * line_height) + (2 * self.padding)

        # Create image
        image = Image.new(
            "RGB",
            (self.max_width, image_height),
            color=self.background_color
        )
        draw = ImageDraw.Draw(image)

        # Draw text line by line
        y_position = self.padding
        for line in lines:
            draw.text(
                (self.padding, y_position),
                line,
                fill=self.text_color,
                font=self.font
            )
            y_position += line_height

        return image

    def _wrap_text(self, text: str) -> list:
        """
        Wrap text to fit within max_width.

        Args:
            text: Text to wrap

        Returns:
            List of text lines
        """
        lines = []
        paragraphs = text.split('\n')

        for paragraph in paragraphs:
            if not paragraph.strip():
                lines.append("")
                continue

            words = paragraph.split()
            current_line = []

            for word in words:
                test_line = ' '.join(current_line + [word])
                # Estimate width (rough approximation)
                estimated_width = len(test_line) * (self.font_size * 0.6)

                if estimated_width <= (self.max_width - 2 * self.padding):
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]

            if current_line:
                lines.append(' '.join(current_line))

        return lines

    def _compress_image(self, image: Image.Image) -> bytes:
        """
        Compress PIL Image to bytes.

        Args:
            image: PIL Image to compress

        Returns:
            Compressed image as bytes
        """
        buffer = io.BytesIO()

        # Use JPEG for good compression while maintaining readability
        image.save(
            buffer,
            format="JPEG",
            quality=self.compression_quality,
            optimize=True
        )

        return buffer.getvalue()

    def get_compression_stats(self, text: str, image_bytes: bytes) -> dict:
        """
        Calculate compression statistics.

        Args:
            text: Original text
            image_bytes: Compressed image bytes

        Returns:
            Dictionary with compression statistics
        """
        text_bytes = len(text.encode('utf-8'))
        image_size = len(image_bytes)

        # Estimate token counts (rough approximation: 1 token ≈ 4 chars)
        text_tokens = len(text) / 4
        # For images, modern vision models use varying token counts
        # For a small text image, we estimate ~10-50 vision tokens
        # This is a simplified estimate
        image_tokens = 20  # Conservative estimate for small text images

        return {
            "text_chars": len(text),
            "text_bytes": text_bytes,
            "image_bytes": image_size,
            "byte_compression_ratio": text_bytes / image_size if image_size > 0 else 0,
            "estimated_text_tokens": text_tokens,
            "estimated_image_tokens": image_tokens,
            "estimated_token_savings": (text_tokens - image_tokens) / text_tokens if text_tokens > 0 else 0,
        }
