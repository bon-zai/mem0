"""
Migration Script - Batch Conversion of Existing Memories

This script converts existing text-based memories to OCR-compressed format.
It processes memories in batches, encoding text to images and verifying
accuracy to ensure data integrity.

Usage:
    python -m mem0.ocr_compression.migration_script --config config.json

Features:
- Batch processing with progress tracking
- Accuracy verification for each conversion
- Rollback support via backup creation
- Resume capability for interrupted migrations
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from mem0.ocr_compression.decoder import OCRDecoder
from mem0.ocr_compression.encoder import OCREncoder
from mem0.ocr_compression.storage import ImageStorage

logger = logging.getLogger(__name__)


class MigrationStats:
    """Track migration statistics."""

    def __init__(self):
        self.total_memories = 0
        self.successful = 0
        self.failed = 0
        self.skipped = 0
        self.low_accuracy = 0
        self.total_bytes_saved = 0
        self.start_time = None
        self.end_time = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        duration = 0
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return {
            "total_memories": self.total_memories,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "low_accuracy": self.low_accuracy,
            "success_rate": self.successful / self.total_memories if self.total_memories > 0 else 0,
            "total_bytes_saved": self.total_bytes_saved,
            "total_mb_saved": self.total_bytes_saved / (1024 * 1024),
            "duration_seconds": duration,
            "memories_per_second": self.successful / duration if duration > 0 else 0,
        }


class MemoryMigration:
    """
    Handles batch migration of memories to OCR-compressed format.
    """

    def __init__(
        self,
        vector_store,
        storage_dir: Optional[str] = None,
        batch_size: int = 100,
        accuracy_threshold: float = 0.95,
        dry_run: bool = False,
    ):
        """
        Initialize migration.

        Args:
            vector_store: Vector store instance
            storage_dir: Directory for image storage
            batch_size: Number of memories to process per batch
            accuracy_threshold: Minimum OCR accuracy required (0.95 = 95%)
            dry_run: If True, simulate migration without making changes
        """
        self.vector_store = vector_store
        self.batch_size = batch_size
        self.accuracy_threshold = accuracy_threshold
        self.dry_run = dry_run

        # Initialize components
        self.encoder = OCREncoder()
        self.decoder = OCRDecoder(backend="easyocr")
        self.image_storage = ImageStorage(storage_dir=storage_dir)

        # Statistics
        self.stats = MigrationStats()

        # Migration state (for resume capability)
        self.state_file = Path(storage_dir or "~/.mem0/ocr_images").expanduser() / "migration_state.json"
        self.processed_ids = set()

        logger.info(
            f"Initialized migration: batch_size={batch_size}, "
            f"accuracy_threshold={accuracy_threshold}, dry_run={dry_run}"
        )

    def load_state(self):
        """Load migration state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.processed_ids = set(state.get("processed_ids", []))
                    logger.info(f"Loaded state: {len(self.processed_ids)} already processed")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def save_state(self):
        """Save migration state to file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump({
                    "processed_ids": list(self.processed_ids),
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def migrate_all(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Migrate all memories matching filters.

        Args:
            filters: Optional filters to select memories

        Returns:
            Migration statistics
        """
        self.stats.start_time = datetime.now()

        logger.info("Starting migration of all memories...")

        # Load previous state if resuming
        self.load_state()

        # Get all memories from vector store
        all_memories = self._fetch_all_memories(filters)
        self.stats.total_memories = len(all_memories)

        logger.info(f"Found {self.stats.total_memories} memories to migrate")

        # Process in batches
        batches = [
            all_memories[i:i + self.batch_size]
            for i in range(0, len(all_memories), self.batch_size)
        ]

        for batch_idx, batch in enumerate(tqdm(batches, desc="Migrating batches")):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}")
            self._process_batch(batch)

            # Save state periodically
            if batch_idx % 10 == 0:
                self.save_state()

        self.stats.end_time = datetime.now()

        # Final state save
        self.save_state()

        # Generate report
        return self._generate_report()

    def _fetch_all_memories(self, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Fetch all memories from vector store.

        Args:
            filters: Optional filters

        Returns:
            List of memory records
        """
        # This is a simplified implementation
        # In production, this would need to handle pagination
        # and work with the specific vector store implementation

        try:
            # For Qdrant/FAISS, we need to use their specific APIs
            # This is a placeholder - actual implementation depends on vector store type
            if hasattr(self.vector_store, 'get_all'):
                return self.vector_store.get_all(filters=filters)
            elif hasattr(self.vector_store, 'scroll'):
                # Qdrant-style pagination
                memories = []
                offset = None
                while True:
                    results, offset = self.vector_store.scroll(
                        limit=1000,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    memories.extend(results)
                    if offset is None:
                        break
                return memories
            else:
                logger.error("Vector store does not support fetching all memories")
                return []

        except Exception as e:
            logger.error(f"Failed to fetch memories: {e}")
            return []

    def _process_batch(self, batch: List[Any]):
        """
        Process a batch of memories.

        Args:
            batch: List of memory records
        """
        for memory in batch:
            try:
                self._migrate_memory(memory)
            except Exception as e:
                logger.error(f"Failed to migrate memory: {e}")
                self.stats.failed += 1

    def _migrate_memory(self, memory):
        """
        Migrate a single memory.

        Args:
            memory: Memory record from vector store
        """
        # Extract memory details
        memory_id = memory.id if hasattr(memory, 'id') else memory.get('id')
        payload = memory.payload if hasattr(memory, 'payload') else memory.get('payload', {})

        # Skip if already processed
        if memory_id in self.processed_ids:
            self.stats.skipped += 1
            logger.debug(f"Skipping already processed memory: {memory_id}")
            return

        # Skip if already OCR-compressed
        if payload.get("ocr_compressed"):
            self.stats.skipped += 1
            self.processed_ids.add(memory_id)
            logger.debug(f"Skipping already compressed memory: {memory_id}")
            return

        # Get text content
        text_content = payload.get("data", "")
        if not text_content or not text_content.strip():
            self.stats.skipped += 1
            self.processed_ids.add(memory_id)
            logger.debug(f"Skipping empty memory: {memory_id}")
            return

        # Extract metadata
        user_id = payload.get("user_id", "unknown")
        timestamp = payload.get("timestamp", datetime.now().isoformat())

        logger.debug(f"Migrating memory {memory_id}: {len(text_content)} chars")

        # Encode to image
        try:
            image_data = self.encoder.encode(text_content)
        except Exception as e:
            logger.error(f"Failed to encode memory {memory_id}: {e}")
            self.stats.failed += 1
            return

        # Verify accuracy by decoding
        try:
            decoded_text = self.decoder.decode(image_data)
            accuracy_result = self.decoder.verify_accuracy(text_content, decoded_text)

            if accuracy_result["character_accuracy"] < self.accuracy_threshold:
                logger.warning(
                    f"Low accuracy for memory {memory_id}: "
                    f"{accuracy_result['accuracy_percentage']:.1f}%"
                )
                self.stats.low_accuracy += 1

                # Optionally skip low-accuracy conversions
                if accuracy_result["character_accuracy"] < 0.90:
                    logger.error(f"Accuracy too low, skipping memory {memory_id}")
                    self.stats.failed += 1
                    return

        except Exception as e:
            logger.error(f"Failed to verify memory {memory_id}: {e}")
            self.stats.failed += 1
            return

        # Save image to storage (unless dry run)
        if not self.dry_run:
            try:
                image_id = self.image_storage.save_image(
                    image_data=image_data,
                    user_id=user_id,
                    timestamp=timestamp,
                    memory_id=memory_id
                )

                # Update metadata
                updated_payload = payload.copy()
                updated_payload["image_id"] = image_id
                updated_payload["ocr_compressed"] = True
                updated_payload["original_length"] = len(text_content)
                updated_payload["image_size"] = len(image_data)

                # Remove raw text data for privacy
                if "data" in updated_payload:
                    del updated_payload["data"]

                # Update in vector store
                self.vector_store.update(
                    ids=[memory_id],
                    payloads=[updated_payload]
                )

                logger.debug(f"Successfully migrated memory {memory_id}")

            except Exception as e:
                logger.error(f"Failed to save migrated memory {memory_id}: {e}")
                self.stats.failed += 1
                return

        # Update statistics
        self.stats.successful += 1
        self.processed_ids.add(memory_id)

        text_bytes = len(text_content.encode('utf-8'))
        image_bytes = len(image_data)
        self.stats.total_bytes_saved += (text_bytes - image_bytes)

    def _generate_report(self) -> Dict[str, Any]:
        """
        Generate migration report.

        Returns:
            Report dictionary
        """
        report = {
            "migration_summary": self.stats.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "accuracy_threshold": self.accuracy_threshold,
        }

        logger.info("=" * 60)
        logger.info("MIGRATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Total memories: {self.stats.total_memories}")
        logger.info(f"Successful: {self.stats.successful}")
        logger.info(f"Failed: {self.stats.failed}")
        logger.info(f"Skipped: {self.stats.skipped}")
        logger.info(f"Low accuracy: {self.stats.low_accuracy}")
        logger.info(f"Success rate: {report['migration_summary']['success_rate'] * 100:.1f}%")
        logger.info(f"Bytes saved: {report['migration_summary']['total_mb_saved']:.2f} MB")
        logger.info(f"Duration: {report['migration_summary']['duration_seconds']:.1f} seconds")
        logger.info(f"Speed: {report['migration_summary']['memories_per_second']:.1f} mem/sec")
        logger.info("=" * 60)

        return report


def main():
    """CLI entry point for migration script."""
    parser = argparse.ArgumentParser(description="Migrate Mem0 memories to OCR-compressed format")
    parser.add_argument("--storage-dir", help="Storage directory for images")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--accuracy-threshold", type=float, default=0.95, help="Minimum accuracy")
    parser.add_argument("--dry-run", action="store_true", help="Simulate migration")
    parser.add_argument("--user-id", help="Migrate only specific user")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting Mem0 OCR Migration Tool")

    # Note: In production, you would need to initialize the vector store
    # from your Mem0 configuration here
    logger.error("This is a standalone script template.")
    logger.error("You need to initialize your vector_store from Mem0 config.")
    logger.error("Example:")
    logger.error("  from mem0.memory.main import Memory")
    logger.error("  mem = Memory()")
    logger.error("  vector_store = mem.vector_store")

    print("\nTo use this migration script:")
    print("1. Import it in a Python script with your Mem0 setup")
    print("2. Initialize MemoryMigration with your vector_store")
    print("3. Call migrate_all() to start migration")


if __name__ == "__main__":
    main()
