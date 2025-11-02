"""
Image Storage - Manages storage and retrieval of image tokens

This module handles the persistent storage of compressed image tokens,
supporting both filesystem and database storage backends.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ImageStorage:
    """
    Manages storage and retrieval of compressed image tokens.

    Supports multiple storage backends:
    - Filesystem (default)
    - Database blob storage (future)
    - Cloud storage (future)
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        storage_backend: str = "filesystem",
    ):
        """
        Initialize image storage.

        Args:
            storage_dir: Directory for storing images (default: ~/.mem0/ocr_images)
            storage_backend: Storage backend to use (default: 'filesystem')
        """
        self.storage_backend = storage_backend

        # Set up storage directory
        if storage_dir is None:
            home_dir = Path.home()
            storage_dir = home_dir / ".mem0" / "ocr_images"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ImageStorage at: {self.storage_dir}")

    def save_image(
        self,
        image_data: bytes,
        user_id: str,
        timestamp: str,
        memory_id: Optional[str] = None
    ) -> str:
        """
        Save image data to storage.

        Args:
            image_data: Image bytes to save
            user_id: User ID for organization
            timestamp: Timestamp for the memory
            memory_id: Optional memory ID

        Returns:
            Image ID or path for later retrieval
        """
        if self.storage_backend == "filesystem":
            return self._save_to_filesystem(image_data, user_id, timestamp, memory_id)
        else:
            raise NotImplementedError(f"Backend '{self.storage_backend}' not implemented")

    def load_image(self, image_id: str) -> bytes:
        """
        Load image data from storage.

        Args:
            image_id: Image ID or path

        Returns:
            Image data as bytes

        Raises:
            FileNotFoundError: If image not found
        """
        if self.storage_backend == "filesystem":
            return self._load_from_filesystem(image_id)
        else:
            raise NotImplementedError(f"Backend '{self.storage_backend}' not implemented")

    def delete_image(self, image_id: str) -> bool:
        """
        Delete image from storage.

        Args:
            image_id: Image ID or path

        Returns:
            True if deleted successfully
        """
        if self.storage_backend == "filesystem":
            return self._delete_from_filesystem(image_id)
        else:
            raise NotImplementedError(f"Backend '{self.storage_backend}' not implemented")

    def _save_to_filesystem(
        self,
        image_data: bytes,
        user_id: str,
        timestamp: str,
        memory_id: Optional[str] = None
    ) -> str:
        """
        Save image to filesystem.

        Args:
            image_data: Image bytes
            user_id: User ID
            timestamp: Timestamp
            memory_id: Optional memory ID

        Returns:
            Relative path to saved image
        """
        # Create user directory
        user_dir = self.storage_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from content hash for deduplication
        content_hash = hashlib.sha256(image_data).hexdigest()[:16]

        # Use memory_id if provided, otherwise use hash-based name
        if memory_id:
            filename = f"{memory_id}_{content_hash}.jpg"
        else:
            # Use timestamp in filename for organization
            timestamp_safe = timestamp.replace(':', '-').replace(' ', '_')
            filename = f"{timestamp_safe}_{content_hash}.jpg"

        filepath = user_dir / filename

        # Save image
        with open(filepath, 'wb') as f:
            f.write(image_data)

        logger.debug(f"Saved image to: {filepath}")

        # Return relative path from storage_dir
        relative_path = filepath.relative_to(self.storage_dir)
        return str(relative_path)

    def _load_from_filesystem(self, image_id: str) -> bytes:
        """
        Load image from filesystem.

        Args:
            image_id: Relative path to image

        Returns:
            Image bytes

        Raises:
            FileNotFoundError: If image not found
        """
        filepath = self.storage_dir / image_id

        if not filepath.exists():
            raise FileNotFoundError(f"Image not found: {image_id}")

        with open(filepath, 'rb') as f:
            image_data = f.read()

        logger.debug(f"Loaded image from: {filepath}")
        return image_data

    def _delete_from_filesystem(self, image_id: str) -> bool:
        """
        Delete image from filesystem.

        Args:
            image_id: Relative path to image

        Returns:
            True if deleted successfully
        """
        filepath = self.storage_dir / image_id

        if filepath.exists():
            filepath.unlink()
            logger.debug(f"Deleted image: {filepath}")
            return True

        logger.warning(f"Image not found for deletion: {image_id}")
        return False

    def get_storage_stats(self) -> dict:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage stats
        """
        if self.storage_backend != "filesystem":
            return {"backend": self.storage_backend, "stats": "unavailable"}

        # Count images and total size
        total_images = 0
        total_size = 0

        for filepath in self.storage_dir.rglob("*.jpg"):
            total_images += 1
            total_size += filepath.stat().st_size

        return {
            "backend": "filesystem",
            "storage_dir": str(self.storage_dir),
            "total_images": total_images,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }

    def cleanup_orphaned_images(self, valid_image_ids: set) -> int:
        """
        Clean up images that are no longer referenced.

        Args:
            valid_image_ids: Set of image IDs that should be kept

        Returns:
            Number of images deleted
        """
        deleted_count = 0

        for filepath in self.storage_dir.rglob("*.jpg"):
            relative_path = str(filepath.relative_to(self.storage_dir))

            if relative_path not in valid_image_ids:
                filepath.unlink()
                deleted_count += 1
                logger.debug(f"Deleted orphaned image: {relative_path}")

        logger.info(f"Cleaned up {deleted_count} orphaned images")
        return deleted_count
