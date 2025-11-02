"""
OCR-based Memory Compression Module

This module provides OCR-based text compression for the Mem0 memory system.
It compresses text memories into visual "image tokens" and decodes them on retrieval,
achieving significant token cost savings while maintaining high accuracy.
"""

from mem0.ocr_compression.encoder import OCREncoder
from mem0.ocr_compression.decoder import OCRDecoder
from mem0.ocr_compression.storage import ImageStorage

__all__ = ["OCREncoder", "OCRDecoder", "ImageStorage"]
