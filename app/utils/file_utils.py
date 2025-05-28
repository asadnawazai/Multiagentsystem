import os
import uuid
import hashlib
from typing import List
from loguru import logger

def create_upload_directory(upload_dir: str = "./uploads") -> str:
    """Create upload directory if it doesn't exist."""
    try:
        os.makedirs(upload_dir, exist_ok=True)
        logger.info(f"Ensured upload directory exists: {upload_dir}")
        return upload_dir
    except Exception as e:
        logger.error(f"Error creating upload directory: {e}")
        raise

def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename to prevent overwrites."""
    try:
        # Get file extension
        extension = original_filename.split('.')[-1] if '.' in original_filename else ''
        
        # Generate a unique identifier
        unique_id = str(uuid.uuid4())
        
        # Combine to create unique filename
        unique_filename = f"{unique_id}.{extension}" if extension else unique_id
        
        return unique_filename
    except Exception as e:
        logger.error(f"Error generating unique filename: {e}")
        raise

def get_file_metadata(file_path: str) -> dict:
    """Get metadata for a file."""
    try:
        file_size_bytes = os.path.getsize(file_path)
        
        # Format file size for display
        if file_size_bytes < 1024:
            file_size_formatted = f"{file_size_bytes} bytes"
        elif file_size_bytes < 1024 * 1024:
            file_size_formatted = f"{file_size_bytes / 1024:.2f} KB"
        else:
            file_size_formatted = f"{file_size_bytes / (1024 * 1024):.2f} MB"
        
        return {
            "file_size_formatted": file_size_formatted,
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
