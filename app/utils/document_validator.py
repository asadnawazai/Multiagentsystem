import os
import re
import yaml
import hashlib
from typing import Dict, List, Tuple, Optional
from loguru import logger


class DocumentValidator:
    """Utility for validating document integrity and real estate document types."""
    
    def __init__(self, config_path: str, uploads_folder: str):
        """Initialize the document validator.
        
        Args:
            config_path: Path to the YAML configuration file
            uploads_folder: Path to the uploads folder where documents are stored
        """
        self.config_path = config_path
        self.uploads_folder = uploads_folder
        self.document_patterns = []
        self.critical_fields = []
        self.checksums = {}
        self._load_config()
        self._load_existing_checksums()
    
    def _load_config(self) -> None:
        """Load the document configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            self.document_patterns = config.get('document_patterns', [])
            self.critical_fields = config.get('critical_fields', [])
            logger.info(f"Loaded document patterns: {self.document_patterns}")
        except Exception as e:
            logger.error(f"Error loading document config: {e}")
            # Set defaults if config loading fails
            self.document_patterns = ["CRS_Property_Report_*", "Zoning_*", "MLS_*"]
            self.critical_fields = ["parcel_id", "tax_value", "property_address"]
    
    def _load_existing_checksums(self) -> None:
        """Load existing checksums from previously processed files."""
        try:
            # Create checksums directory if it doesn't exist
            checksums_path = os.path.join(self.uploads_folder, "checksums")
            os.makedirs(checksums_path, exist_ok=True)
            
            # Load existing checksums from a file if it exists
            checksums_file = os.path.join(checksums_path, "file_checksums.txt")
            
            if os.path.exists(checksums_file):
                with open(checksums_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('|')
                        if len(parts) >= 3:
                            filename, file_size, checksum = parts[:3]
                            self.checksums[filename] = {'size': file_size, 'checksum': checksum}
                            
            logger.info(f"Loaded {len(self.checksums)} existing checksums")
        except Exception as e:
            logger.error(f"Error loading existing checksums: {e}")
    
    def _save_checksum(self, filename: str, file_size: str, checksum: str) -> None:
        """Save a new file checksum to the checksums database."""
        try:
            checksums_path = os.path.join(self.uploads_folder, "checksums")
            checksums_file = os.path.join(checksums_path, "file_checksums.txt")
            
            with open(checksums_file, 'a') as f:
                f.write(f"{filename}|{file_size}|{checksum}\n")
                
            # Update in-memory store
            self.checksums[filename] = {'size': file_size, 'checksum': checksum}
        except Exception as e:
            logger.error(f"Error saving checksum: {e}")
    
    def is_valid_real_estate_document(self, filename: str) -> bool:
        """Check if file is a valid real estate document based on filename patterns.
        
        Args:
            filename: The original filename to check
            
        Returns:
            bool: True if the file matches any valid real estate document pattern
        """
        # Check each pattern for a match
        for pattern in self.document_patterns:
            # Convert glob pattern to regex pattern
            regex_pattern = pattern.replace('*', '.*')
            if re.match(regex_pattern, filename, re.IGNORECASE):
                logger.info(f"File {filename} matches pattern {pattern}")
                return True
                
        logger.warning(f"File {filename} does not match any real estate document patterns")
        return False
    
    def validate_file_integrity(self, filepath: str, original_filename: str) -> Tuple[bool, Optional[str]]:
        """Validate file integrity by checking for empty files, corrupted files, and duplicates.
        
        Args:
            filepath: Path to the file
            original_filename: Original filename from upload
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                return False, "File does not exist"
                
            # Check if file is empty
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                return False, "File is empty"
                
            # Calculate checksum
            checksum = self._calculate_checksum(filepath)
            
            # UPDATED LOGIC: First check if the document exists in the database
            # Only consider a file a duplicate if it exists in the database
            if self._is_duplicate_in_database(checksum):
                return False, "This document has already been processed and exists in the database."
            
            # The file may have been uploaded before but doesn't exist in the database anymore
            # (e.g., it was deleted) - in this case, we allow it to be processed again
            
            # File is valid - save its checksum locally for reference
            self._save_checksum(original_filename, str(file_size), checksum)
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating file integrity: {e}")
            return False, f"Error validating file: {str(e)}"
    
    def _calculate_checksum(self, filepath: str) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        
        with open(filepath, "rb") as f:
            # Read and update hash in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def _is_duplicate_in_database(self, checksum: str) -> bool:
        """Check if a document with the same checksum already exists in the database.
        
        Args:
            checksum: The SHA-256 checksum of the file
            
        Returns:
            bool: True if a document with the same checksum exists in the database
        """
        try:
            import psycopg2
            import os
            
            # Get database connection parameters from environment variables
            # Try PG_* prefix first, then fall back to DB_* prefix
            host = os.getenv('PG_HOST', os.getenv('DB_HOST', 'localhost'))
            port = os.getenv('PG_PORT', os.getenv('DB_PORT', '5432'))
            dbname = os.getenv('PG_DATABASE', os.getenv('DB_NAME', 'panoramascore'))
            user = os.getenv('PG_USER', os.getenv('DB_USER', 'postgres'))
            password = os.getenv('PG_PASSWORD', os.getenv('DB_PASSWORD', ''))
            schema = os.getenv('PG_SCHEMA', 'public')
            
            conn = psycopg2.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password
            )
            
            with conn.cursor() as cursor:
                # Query for documents with the same checksum
                cursor.execute("SELECT id FROM real_estate_documents WHERE file_checksum = %s", (checksum,))
                result = cursor.fetchone()
                
            conn.close()
            
            # If we found a result, a duplicate exists
            return result is not None
            
        except Exception as e:
            # If there's an error connecting to the database, log it but don't block the upload
            logger.error(f"Error checking for duplicates in database: {e}")
            return False
    
    def check_missing_critical_fields(self, fields: Dict) -> List[str]:
        """Check if any critical fields are missing from the extracted data.
        
        Args:
            fields: Dictionary of extracted fields
            
        Returns:
            List[str]: List of missing critical fields
        """
        missing_fields = []
        
        for field in self.critical_fields:
            if field not in fields or not fields[field]:
                missing_fields.append(field)
                
        if missing_fields:
            logger.warning(f"Missing critical fields: {missing_fields}")
            
        return missing_fields
