from loguru import logger
from typing import Dict, Optional
from ..utils.file_utils import get_file_metadata, calculate_checksum


class MetadataHashingAgent:
    """Agent responsible for generating file metadata and checksums."""
    
    def __init__(self):
        pass
    
    def process_document(self, file_path: str, original_filename: str, client_id: Optional[str] = None) -> Dict:
        """Extract metadata and generate hash for a document.
        
        Args:
            file_path: Path to the saved file
            original_filename: Original filename from upload
            client_id: Optional client identifier
            
        Returns:
            Dict: Document metadata including filename, upload time, file size, checksum
        """
        try:
            # Get basic file metadata
            metadata = get_file_metadata(file_path)
            
            # Use original filename in the response instead of the unique generated one
            metadata["filename"] = original_filename
            
            # Calculate checksum
            checksum = calculate_checksum(file_path)
            metadata["checksum"] = checksum
            
            # Add client ID if provided
            if client_id:
                metadata["client_id"] = client_id
                
            logger.info(f"Generated metadata for {original_filename}")
            return metadata
        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            raise
