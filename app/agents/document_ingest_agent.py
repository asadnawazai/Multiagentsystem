import os
import shutil
import re
import asyncio
import mimetypes
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any
from loguru import logger
from fastapi import UploadFile
from ..utils.file_utils import is_valid_file_extension, generate_unique_filename, create_upload_directory, calculate_checksum
from ..utils.document_validator import DocumentValidator
from ..utils.document_type_detector import DocumentTypeDetector
from ..utils.yaml_adapter import YAMLAdapter
from .ocr_normalization_agent import OCRNormalizationAgent


class DocumentIngestAgent:
    """Agent responsible for ingesting and saving uploaded documents.
    
    Implements validation for real estate documents, document type detection,
    OCR processing, and field extraction.
    """
    
    def __init__(self, upload_folder: str, allowed_extensions: list, max_file_size_mb: int):
        self.upload_folder = upload_folder
        self.allowed_extensions = allowed_extensions
        self.max_file_size_mb = max_file_size_mb
        create_upload_directory(upload_folder)
        
        # Create logs directory
        self.logs_folder = os.path.join('app', 'logs', 'ocr_logs')
        os.makedirs(self.logs_folder, exist_ok=True)
        
        # Initialize document validator for real estate documents
        config_path = os.path.join('app', 'config', 'real_estate.yaml')
        self.document_validator = DocumentValidator(config_path, upload_folder)
        
        # Initialize document type detector
        self.document_detector = DocumentTypeDetector()
        
        # Initialize OCR agent with configuration
        ocr_config = {
            'tesseract_config': '--psm 6 -l eng',
            'log_dir': self.logs_folder
        }
        self.ocr_agent = OCRNormalizationAgent(ocr_config=ocr_config)
        
        # Initialize YAML adapter for scoring configuration
        self.yaml_adapter = YAMLAdapter(config_path)
    
    async def ingest_document(self, file: UploadFile, document_type: str = "Real Estate", client_id: Optional[str] = None) -> Tuple[str, str, Dict]:
        """Process an uploaded file and save it to the upload folder.
        
        Args:
            file: The uploaded file object
            document_type: Type of document being processed (default: Real Estate)
            client_id: Optional client identifier
            
        Returns:
            Tuple[str, str, Dict]: The path to the saved file, original filename, and validation info
            
        Raises:
            ValueError: If file type is not allowed, file is too large, or not a valid real estate document
        """
        validation_info = {
            "is_valid": True,
            "message": "",
            "is_real_estate_doc": True,
            "upload_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": file.filename,
            "mime_type": file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
        }
        
        # Validate file extension
        if not is_valid_file_extension(file.filename, self.allowed_extensions):
            logger.warning(f"Invalid file extension: {file.filename}")
            raise ValueError(f"File type not allowed. Allowed types: {', '.join(self.allowed_extensions)}")
        
        # Check if it's a valid document type based on filename, but don't reject it
        is_valid_pattern = self.document_validator.is_valid_real_estate_document(file.filename)
        if not is_valid_pattern:
            logger.warning(f"File does not match standard naming patterns: {file.filename}")
            # Add a warning but continue processing
            validation_info["filename_warning"] = f"File naming doesn't match standard patterns like {', '.join(self.document_validator.document_patterns)}"

        # Read file content to check size
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)  # Convert bytes to MB
        
        # Validate file size
        if file_size_mb > self.max_file_size_mb:
            logger.warning(f"File too large: {file_size_mb:.2f}MB (max: {self.max_file_size_mb}MB)")
            raise ValueError(f"File too large. Maximum size: {self.max_file_size_mb}MB")
        
        # Check if file is empty
        if len(file_content) == 0:
            validation_info["is_valid"] = False
            validation_info["message"] = "The uploaded file is empty."
            logger.warning(f"Empty file uploaded: {file.filename}")
            raise ValueError("The uploaded file is empty")
            
        # Generate a unique filename
        unique_filename = generate_unique_filename(file.filename)
        file_path = os.path.join(self.upload_folder, unique_filename)
        
        # Write the file to disk
        try:
            with open(file_path, "wb") as f:
                f.write(file_content)
            logger.info(f"File saved: {file_path}")
            
            # Get file stats for metadata
            file_stats = os.stat(file_path)
            file_size_bytes = file_stats.st_size
            file_size_kb = file_size_bytes / 1024
            file_size_mb = file_size_kb / 1024
            
            # Add file size metadata
            validation_info["file_size_bytes"] = file_size_bytes
            validation_info["file_size_formatted"] = f"{file_size_mb:.2f} MB" if file_size_mb >= 1 else f"{file_size_kb:.2f} KB"
            
            # Ensure MIME type is correctly identified
            if not validation_info["mime_type"] or validation_info["mime_type"] == "application/octet-stream":
                mime_type = mimetypes.guess_type(file_path)[0]
                if mime_type:
                    validation_info["mime_type"] = mime_type
            
            # Validate file integrity and check for duplicates
            is_valid, error_message = self.document_validator.validate_file_integrity(file_path, file.filename)
            
            # Calculate and store checksum regardless of validation result
            checksum = self.document_validator._calculate_checksum(file_path)
            validation_info["file_checksum"] = checksum
            
            if not is_valid:
                validation_info["is_valid"] = False
                validation_info["message"] = error_message
                logger.warning(f"File integrity check failed: {error_message}")
            
            # Detect document type (digital or scanned)
            try:
                document_info = self.document_detector.detect_document_type(file_path)
                document_type_result = document_info.get('document_type', 'unknown')
                needs_ocr = document_info.get('needs_ocr', False)
                rotation_angle = document_info.get('rotation_angle', 0)
                
                # Add document type info to validation_info
                validation_info["document_type"] = document_type_result
                validation_info["needs_ocr"] = needs_ocr
                validation_info["rotation_angle"] = rotation_angle
                
                logger.info(f"Document type detected: {document_type_result}, needs OCR: {needs_ocr}, rotation: {rotation_angle}°")
                
                # Process document with OCR if needed
                ocr_result = await self._process_document_with_ocr(file_path)
                
                # Add OCR results to validation_info
                validation_info.update({
                    "ocr_processed": True,
                    "extraction_method": ocr_result.get("extraction_method", "unknown"),
                    "ocr_confidence": ocr_result.get("confidence"),
                    "extracted_fields": ocr_result.get("extracted_fields", {}),
                    "missing_fields": ocr_result.get("missing_fields", [])
                })
                
                # Update YAML adapter with extracted fields
                if ocr_result.get("extracted_fields"):
                    self.yaml_adapter.update_fields(ocr_result["extracted_fields"])
                    logger.info(f"Updated YAML config with {len(ocr_result['extracted_fields'])} extracted fields")
                
            except Exception as e:
                logger.error(f"Error during document type detection or OCR: {e}")
                validation_info["ocr_error"] = str(e)
            
            return file_path, file.filename, validation_info
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise
    
    async def _process_document_with_ocr(self, file_path: str) -> Dict[str, Any]:
        """Process a document with OCR to extract text and fields.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict containing OCR results and extracted fields
        """
        try:
            # Process the document with OCR
            ocr_result = await self.ocr_agent.process_document(file_path)
            
            # Return the OCR results
            return ocr_result
        except Exception as e:
            logger.error(f"Error processing document with OCR: {e}")
            return {
                "error": str(e),
                "extraction_method": "failed",
                "extracted_fields": {},
                "missing_fields": []
            }
