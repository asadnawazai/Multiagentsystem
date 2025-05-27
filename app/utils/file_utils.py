import os
import hashlib
import uuid
import datetime
from pathlib import Path
from typing import Optional, Tuple, List
from loguru import logger


def get_file_metadata(file_path: str) -> dict:
    """Extract metadata from a file."""
    try:
        file_stats = os.stat(file_path)
        file_size_bytes = file_stats.st_size
        file_size_kb = round(file_size_bytes / 1024, 2)
        file_size = f"{file_size_kb}KB"
        
        # Get the filename without path
        filename = os.path.basename(file_path)
        
        # Get the current time in ISO format
        upload_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        return {
            "filename": filename,
            "upload_time": upload_time,
            "file_size": file_size,
            "file_size_bytes": file_size_bytes
        }
    except Exception as e:
        logger.error(f"Error getting file metadata: {e}")
        raise


def calculate_checksum(file_path: str) -> str:
    """Calculate SHA-256 checksum of a file."""
    try:
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read and update hash in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating checksum: {e}")
        raise


def is_valid_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Check if file has an allowed extension."""
    extension = filename.split('.')[-1].lower()
    return extension in allowed_extensions


def create_upload_directory(upload_dir: str) -> None:
    """Create upload directory if it doesn't exist."""
    try:
        Path(upload_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured upload directory exists: {upload_dir}")
    except Exception as e:
        logger.error(f"Failed to create upload directory {upload_dir}: {e}")
        raise


def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename while preserving the original extension."""
    extension = original_filename.split('.')[-1]
    unique_id = str(uuid.uuid4())
    return f"{unique_id}.{extension}"
