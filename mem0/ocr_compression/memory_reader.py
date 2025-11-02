"""
Memory Reader - OCR-Integrated Memory Retrieval Pipeline

This module enables querying the memory store and retrieving original text from
image tokens. It modifies Mem0's retrieval flow to include an OCR decode step
after vector search.

Key steps:
1. Accept natural language query
2. Compute semantic embedding of query
3. Perform vector similarity search
4. For OCR-compressed memories, decode image tokens back to text
5. Return results with reconstructed content
"""

import logging
from typing import Any, Dict, List, Optional

from mem0.ocr_compression.decoder import OCRDecoder
from mem0.ocr_compression.storage import ImageStorage

logger = logging.getLogger(__name__)


class MemoryReader:
    """
    OCR-integrated memory reader.

    This class handles querying and retrieval of memories, with support for
    decoding OCR-compressed image tokens back to text.
    """

    def __init__(
        self,
        vector_store,
        embedding_model,
        storage_dir: Optional[str] = None,
        ocr_backend: str = "easyocr",
        decoder_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the memory reader.

        Args:
            vector_store: Vector store instance (FAISS, Qdrant, etc.)
            embedding_model: Embedding model for query vectors
            storage_dir: Directory where image tokens are stored
            ocr_backend: OCR backend to use ('easyocr', 'paddleocr', 'tesseract')
            decoder_config: Configuration for OCR decoder
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model

        # Initialize OCR components
        decoder_config = decoder_config or {}
        if "backend" not in decoder_config:
            decoder_config["backend"] = ocr_backend

        try:
            self.decoder = OCRDecoder(**decoder_config)
            self.image_storage = ImageStorage(storage_dir=storage_dir)
            logger.info(f"Initialized OCR decoder with backend: {ocr_backend}")
        except Exception as e:
            logger.warning(f"Failed to initialize OCR decoder: {e}")
            logger.warning("OCR-compressed memories will not be decodable")
            self.decoder = None
            self.image_storage = None

    def query_memory(
        self,
        query: str,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Query memories and decode OCR-compressed results.

        Args:
            query: Natural language query
            user_id: Filter by user ID
            filters: Additional metadata filters
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            Dictionary with query results
        """
        # Step 1: Embed the query for semantic search
        logger.debug(f"Embedding query: {query[:100]}...")
        query_embeddings = self.embedding_model.embed(query, memory_action="search")

        # Step 2: Build filters
        search_filters = filters.copy() if filters else {}
        if user_id:
            search_filters["user_id"] = user_id

        # Step 3: Perform vector similarity search
        logger.debug(f"Searching vector store with filters: {search_filters}")
        search_results = self.vector_store.search(
            query=query,
            vectors=query_embeddings,
            limit=top_k,
            filters=search_filters,
        )

        # Step 4: Decode OCR-compressed memories
        decoded_memories = []
        for result in search_results:
            decoded_memory = self._decode_memory_result(result, threshold)
            if decoded_memory:
                decoded_memories.append(decoded_memory)

        logger.info(f"Retrieved {len(decoded_memories)} memories for query")

        return {
            "query": query,
            "results": decoded_memories,
            "count": len(decoded_memories),
        }

    def _decode_memory_result(
        self,
        result,
        threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Decode a single memory result.

        Args:
            result: Search result from vector store
            threshold: Optional similarity threshold

        Returns:
            Decoded memory dictionary or None if below threshold
        """
        # Check similarity threshold
        if threshold is not None and hasattr(result, 'score'):
            if result.score < threshold:
                return None

        # Extract metadata
        metadata = result.payload if hasattr(result, 'payload') else {}
        memory_id = result.id if hasattr(result, 'id') else metadata.get('id', 'unknown')

        # Check if memory is OCR-compressed
        is_ocr_compressed = metadata.get("ocr_compressed", False)

        if is_ocr_compressed:
            # Decode from image token
            memory_text = self._decode_from_image(metadata, memory_id)
        else:
            # Get from traditional text storage
            memory_text = metadata.get("data", "")

        # Build result dictionary
        decoded_memory = {
            "id": memory_id,
            "memory": memory_text,
            "user_id": metadata.get("user_id"),
            "timestamp": metadata.get("timestamp"),
            "created_at": metadata.get("created_at"),
            "ocr_compressed": is_ocr_compressed,
        }

        # Add score if available
        if hasattr(result, 'score'):
            decoded_memory["score"] = result.score

        # Add any custom metadata (excluding internal fields)
        excluded_fields = {
            "data", "image_id", "ocr_compressed", "hash", "original_length",
            "image_size", "compression_ratio", "user_id", "timestamp", "created_at"
        }
        for key, value in metadata.items():
            if key not in excluded_fields and key not in decoded_memory:
                decoded_memory[key] = value

        return decoded_memory

    def _decode_from_image(
        self,
        metadata: Dict[str, Any],
        memory_id: str,
    ) -> str:
        """
        Decode text from stored image token.

        Args:
            metadata: Memory metadata containing image reference
            memory_id: Memory ID for logging

        Returns:
            Decoded text content
        """
        image_id = metadata.get("image_id")

        if not image_id:
            logger.error(f"Memory {memory_id} marked as OCR-compressed but missing image_id")
            return "<OCR Error: Missing image reference>"

        if not self.decoder or not self.image_storage:
            logger.error(f"OCR decoder not available for memory {memory_id}")
            return "<OCR Error: Decoder not available>"

        try:
            # Load image from storage
            logger.debug(f"Loading image token: {image_id}")
            image_data = self.image_storage.load_image(image_id)

            # Decode image to text using OCR
            logger.debug(f"Decoding image token for memory {memory_id}")
            decoded_text = self.decoder.decode(image_data)

            return decoded_text

        except FileNotFoundError:
            logger.error(f"Image token not found: {image_id} for memory {memory_id}")
            return "<OCR Error: Image not found>"

        except Exception as e:
            logger.error(f"Failed to decode memory {memory_id}: {e}")
            return f"<OCR Error: {str(e)}>"

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: Memory ID to retrieve

        Returns:
            Memory dictionary or None if not found
        """
        try:
            # Get from vector store
            result = self.vector_store.get(memory_id)

            if result:
                return self._decode_memory_result(result, threshold=None)

            return None

        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None

    def verify_ocr_accuracy(
        self,
        original_text: str,
        memory_id: str,
    ) -> Dict[str, Any]:
        """
        Verify OCR decoding accuracy for a memory.

        Args:
            original_text: Original text before encoding
            memory_id: Memory ID to verify

        Returns:
            Accuracy verification results
        """
        # Retrieve the memory
        memory = self.get_memory_by_id(memory_id)

        if not memory:
            return {
                "error": "Memory not found",
                "memory_id": memory_id,
            }

        if not memory.get("ocr_compressed"):
            return {
                "error": "Memory is not OCR-compressed",
                "memory_id": memory_id,
            }

        decoded_text = memory.get("memory", "")

        # Verify accuracy
        if self.decoder:
            accuracy_results = self.decoder.verify_accuracy(original_text, decoded_text)
            accuracy_results["memory_id"] = memory_id
            return accuracy_results
        else:
            return {
                "error": "OCR decoder not available",
                "memory_id": memory_id,
            }

    def get_reader_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory reader.

        Returns:
            Dictionary with reader statistics
        """
        stats = {
            "decoder_available": self.decoder is not None,
            "decoder_backend": self.decoder.backend if self.decoder else None,
        }

        if self.image_storage:
            storage_stats = self.image_storage.get_storage_stats()
            stats.update(storage_stats)

        return stats
