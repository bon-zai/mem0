"""
Memory Writer - OCR-Integrated Memory Write Pipeline

This module handles writing new memories with OCR compression. It extends the Mem0
writing flow to encode text content into compressed image tokens and store them
along with vector embeddings and metadata.

Key steps:
1. Accept memory input (user_id, timestamp, content, metadata)
2. Encode content to image token using OCR encoder
3. Store image token to persistent storage
4. Compute text embedding for semantic search
5. Upsert into vector store with image reference in metadata
"""

import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import pytz

from mem0.ocr_compression.encoder import OCREncoder
from mem0.ocr_compression.storage import ImageStorage

logger = logging.getLogger(__name__)


class MemoryWriter:
    """
    OCR-integrated memory writer.

    This class handles the creation of new memories with OCR-based text compression.
    It integrates with Mem0's existing vector store and embedding infrastructure
    while adding OCR compression for content storage.
    """

    def __init__(
        self,
        vector_store,
        embedding_model,
        storage_dir: Optional[str] = None,
        enable_ocr_compression: bool = True,
        encoder_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the memory writer.

        Args:
            vector_store: Vector store instance (FAISS, Qdrant, etc.)
            embedding_model: Embedding model for creating vectors
            storage_dir: Directory for storing image tokens
            enable_ocr_compression: Whether to enable OCR compression (default: True)
            encoder_config: Configuration for OCR encoder
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.enable_ocr_compression = enable_ocr_compression

        # Initialize OCR components
        if self.enable_ocr_compression:
            encoder_config = encoder_config or {}
            self.encoder = OCREncoder(**encoder_config)
            self.image_storage = ImageStorage(storage_dir=storage_dir)
            logger.info("OCR compression enabled for memory writer")
        else:
            self.encoder = None
            self.image_storage = None
            logger.info("OCR compression disabled - using traditional text storage")

    def write_memory(
        self,
        content: str,
        user_id: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Write a new memory with optional OCR compression.

        Args:
            content: The text content to store
            user_id: User ID for the memory
            timestamp: Timestamp (default: current time)
            metadata: Additional metadata to store
            memory_id: Optional memory ID (default: auto-generated)

        Returns:
            Dictionary with status and memory details
        """
        # Generate defaults
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        if timestamp is None:
            timestamp = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        # Initialize metadata
        record_metadata = metadata.copy() if metadata else {}
        record_metadata["user_id"] = user_id
        record_metadata["timestamp"] = timestamp
        record_metadata["created_at"] = timestamp
        record_metadata["hash"] = hashlib.md5(content.encode()).hexdigest()

        # Compute embedding for semantic search
        embeddings = self.embedding_model.embed(content, memory_action="add")

        # Handle OCR compression or traditional storage
        if self.enable_ocr_compression and self.encoder and self.image_storage:
            result = self._write_with_ocr(
                content=content,
                memory_id=memory_id,
                user_id=user_id,
                timestamp=timestamp,
                embeddings=embeddings,
                metadata=record_metadata,
            )
        else:
            result = self._write_traditional(
                content=content,
                memory_id=memory_id,
                embeddings=embeddings,
                metadata=record_metadata,
            )

        return result

    def _write_with_ocr(
        self,
        content: str,
        memory_id: str,
        user_id: str,
        timestamp: str,
        embeddings,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Write memory with OCR compression.

        Args:
            content: Text content
            memory_id: Memory ID
            user_id: User ID
            timestamp: Timestamp
            embeddings: Vector embeddings
            metadata: Metadata dict

        Returns:
            Result dictionary
        """
        try:
            # Step 1: Encode content to image token
            logger.debug(f"Encoding content to image token (length: {len(content)} chars)")
            image_data = self.encoder.encode(content)

            # Step 2: Store image token
            image_id = self.image_storage.save_image(
                image_data=image_data,
                user_id=user_id,
                timestamp=timestamp,
                memory_id=memory_id
            )

            logger.debug(f"Stored image token: {image_id}")

            # Step 3: Update metadata with image reference
            metadata["image_id"] = image_id
            metadata["ocr_compressed"] = True
            metadata["original_length"] = len(content)
            metadata["image_size"] = len(image_data)

            # Get compression stats
            stats = self.encoder.get_compression_stats(content, image_data)
            metadata["compression_ratio"] = stats.get("byte_compression_ratio", 0)

            # Step 4: Store original content hash for verification (not the content itself)
            # We don't store "data" field to maintain privacy
            # The content will be reconstructed via OCR on retrieval

            # Step 5: Insert into vector store
            self.vector_store.insert(
                vectors=[embeddings],
                ids=[memory_id],
                payloads=[metadata],
            )

            logger.info(
                f"Wrote OCR-compressed memory {memory_id}: "
                f"{len(content)} chars -> {len(image_data)} bytes "
                f"(compression: {stats.get('byte_compression_ratio', 0):.2f}x)"
            )

            return {
                "status": "success",
                "memory_id": memory_id,
                "image_id": image_id,
                "ocr_compressed": True,
                "compression_stats": stats,
            }

        except Exception as e:
            logger.error(f"Failed to write OCR-compressed memory: {e}")
            # Fallback to traditional storage
            logger.warning("Falling back to traditional text storage")
            return self._write_traditional(content, memory_id, embeddings, metadata)

    def _write_traditional(
        self,
        content: str,
        memory_id: str,
        embeddings,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Write memory with traditional text storage (fallback).

        Args:
            content: Text content
            memory_id: Memory ID
            embeddings: Vector embeddings
            metadata: Metadata dict

        Returns:
            Result dictionary
        """
        # Store content directly in metadata (traditional Mem0 approach)
        metadata["data"] = content
        metadata["ocr_compressed"] = False

        # Insert into vector store
        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )

        logger.debug(f"Wrote traditional memory {memory_id}")

        return {
            "status": "success",
            "memory_id": memory_id,
            "ocr_compressed": False,
        }

    def get_writer_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory writer.

        Returns:
            Dictionary with writer statistics
        """
        stats = {
            "ocr_compression_enabled": self.enable_ocr_compression,
        }

        if self.enable_ocr_compression and self.image_storage:
            storage_stats = self.image_storage.get_storage_stats()
            stats.update(storage_stats)

        return stats
