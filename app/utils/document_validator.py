import os
import re
import hashlib
import psycopg2
from typing import List, Dict, Optional, Tuple, Any
from loguru import logger
from app.utils.yaml_adapter import YAMLAdapter

# Database connection settings - using default values if not in environment
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DATABASE = os.getenv("PG_DATABASE", "postgres")

class DocumentValidator:
    """Validator for real estate documents.
    
    This class validates documents based on configured patterns for real estate documents
    and checks for duplicates in the database by using file checksums.
    """
    
    def __init__(self, config_path: str, upload_folder: str):
        """Initialize the document validator.
        
        Args:
            config_path: Path to the configuration file
            upload_folder: Path to the upload folder
        """
        self.config_path = config_path
        self.upload_folder = upload_folder
        self.yaml_adapter = YAMLAdapter(config_path)
        
        # Load document patterns and validation rules
        self.document_patterns = self._load_config()
        
        # Load existing checksums to avoid duplicates
        self.existing_checksums = self._load_existing_checksums()
        
    def _load_config(self) -> List[str]:
        """Load document patterns from configuration.
        
        Returns:
            List of document patterns
        """
        try:
            config = self.yaml_adapter.load_config()
            if 'document_patterns' in config:
                patterns = config['document_patterns']
                logger.info(f"Loaded document patterns: {patterns}")
                return patterns
            else:
                # Default patterns if none provided
                default_patterns = [
                    "*.pdf", "*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff",
                    "*Property*", "*Tax*", "*Deed*", "*Title*", "*Survey*", "*Mortgage*",
                    "*Closing*", "*Listing*", "*Estate*", "*House*", "*Home*", "*Parcel*",
                    "Property*", "Tax*", "Deed*", "Title*", "Survey*", "Mortgage*",
                    "Closing*", "Listing*", "Real_Estate*", "House*", "Home*", "Parcel*",
                    "MLS_*", "Property_Tax_*", "Title_Search_*", "Flood_Map_*", 
                    "Zoning_*", "Appraisal_*", "CRS_Property_Report_*", "Assessment*"
                ]
                logger.warning(f"No document patterns found in config, using defaults: {default_patterns}")
                return default_patterns
        except Exception as e:
            logger.error(f"Error loading document patterns: {e}")
            # Return default patterns in case of error
            return ["*.pdf", "*.jpg", "*.jpeg", "*.png"]
            
    def _load_existing_checksums(self) -> List[str]:
        """Load existing file checksums from the database.
        
        Returns:
            List of existing checksums
        """
        try:
            conn = psycopg2.connect(
                host=PG_HOST,
                port=PG_PORT,
                user=PG_USER,
                password=PG_PASSWORD,
                database=PG_DATABASE
            )
            
            with conn.cursor() as cursor:
                cursor.execute("SELECT file_checksum FROM real_estate_documents")
                checksums = [row[0] for row in cursor.fetchall()]
                
            conn.close()
            logger.info(f"Loaded {len(checksums)} existing checksums")
            return checksums
        except Exception as e:
            logger.error(f"Error loading existing checksums: {e}")
            return []
            
    def is_valid_real_estate_document(self, filename: str) -> bool:
        """Check if the file is a valid real estate document.
        
        Args:
            filename: Name of the file to validate
            
        Returns:
            True if the file is a valid real estate document, False otherwise
        """
        # First check if the file exists
        if not os.path.exists(filename):
            return False
            
        # Get the base filename without path
        base_filename = os.path.basename(filename)
        
        # Check if the file matches any of the document patterns
        for pattern in self.document_patterns:
            if self._matches_pattern(base_filename, pattern):
                logger.info(f"File {base_filename} matches pattern {pattern}")
                return True
                
        logger.warning(f"File {base_filename} does not match any document pattern")
        return False
        
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if the filename matches the pattern.
        
        Args:
            filename: Name of the file to validate
            pattern: Pattern to match against
            
        Returns:
            True if the file matches the pattern, False otherwise
        """
        # Convert the glob pattern to regex pattern
        regex_pattern = pattern.replace('.', '\\.')
        regex_pattern = regex_pattern.replace('*', '.*')
        regex_pattern = f"^{regex_pattern}$"
        
        return bool(re.match(regex_pattern, filename, re.IGNORECASE))
        
    def is_duplicate(self, file_path: str) -> bool:
        """Check if the file is a duplicate.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file is a duplicate, False otherwise
        """
        # Calculate checksum for the file
        with open(file_path, 'rb') as f:
            file_content = f.read()
            checksum = hashlib.sha256(file_content).hexdigest()
            
        # Check if the checksum exists in the database
        return self._is_duplicate_in_database(checksum)
        
    def _is_duplicate_in_database(self, checksum: str) -> bool:
        """Check if the checksum exists in the database.
        
        Args:
            checksum: Checksum to check
            
        Returns:
            True if the checksum exists in the database, False otherwise
        """
        try:
            conn = psycopg2.connect(
                host=PG_HOST,
                port=PG_PORT,
                user=PG_USER,
                password=PG_PASSWORD,
                database=PG_DATABASE
            )
            
            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM real_estate_documents WHERE file_checksum = %s", (checksum,))
                result = cursor.fetchone()
                
            conn.close()
            return result is not None
        except Exception as e:
            logger.error(f"Error checking for duplicate in database: {e}")
            return False
            
    def get_document_checksum(self, file_path: str) -> str:
        """Calculate the checksum for a document.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Checksum of the file
        """
        with open(file_path, 'rb') as f:
            file_content = f.read()
            return hashlib.sha256(file_content).hexdigest()
            
    def _calculate_checksum(self, file_path: str) -> str:
        """Private method to calculate document checksum.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Checksum of the file
        """
        # This is an alias for get_document_checksum to maintain compatibility
        return self.get_document_checksum(file_path)
            
    def validate_file_integrity(self, file_path: str, original_filename: str) -> Tuple[bool, str]:
        """Validate the integrity of a document.
        
        Args:
            file_path: Path to the file
            original_filename: Original name of the file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if the file exists
        if not os.path.exists(file_path):
            return False, f"File does not exist: {file_path}"
            
        # Check if the file is a valid real estate document
        if not self.is_valid_real_estate_document(original_filename):
            # If strict validation is required, return False
            # For now, we'll allow it but log a warning
            logger.warning(f"File does not match standard naming patterns: {original_filename}")
            
        # Check if the file is a duplicate
        file_checksum = self.get_document_checksum(file_path)
        if file_checksum in self.existing_checksums:
            return False, f"Duplicate file detected with checksum: {file_checksum}"
            
        # File passed all integrity checks
        return True, "File integrity validated"
