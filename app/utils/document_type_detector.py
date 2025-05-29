import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

class DocumentTypeDetector:
    """Utility for detecting document types and properties.
    
    This class provides functionality to detect document types,
    determine if OCR is needed, and detect rotation angles.
    """
    
    def __init__(self):
        """Initialize the document type detector."""
        # Document type patterns
        self.document_patterns = {
            'real_estate': [
                r'(?i)real\s+estate',
                r'(?i)property',
                r'(?i)appraisal',
                r'(?i)zoning',
                r'(?i)MLS',
                r'(?i)listing',
                r'(?i)flood\s+map',
                r'(?i)title\s+search'
            ]
        }
        
    def detect_document_type(self, file_path: str) -> Dict[str, Any]:
        """Detect document type and properties from a file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict with document type, OCR needs, and rotation information
        """
        result = {
            'document_type': 'Unknown',
            'needs_ocr': False,
            'rotation_angle': 0
        }
        
        try:
            file_ext = Path(file_path).suffix.lower()
            
            # Check if it's a PDF document
            if file_ext == '.pdf':
                result['document_type'] = 'PDF Document'
                # For PDFs, determine if it needs OCR (scanned vs digital)
                result['needs_ocr'] = True  # Simplified for now
                
            # Check if it's an image document (likely needs OCR)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']:
                result['document_type'] = 'Image Document'
                result['needs_ocr'] = True
                
            # For specific document classification, use filename hints
            document_type = self._classify_document_content(file_path)
            if document_type:
                result['document_type'] = document_type
                
        except Exception as e:
            logger.error(f"Error detecting document type: {e}")
            
        return result
    
    def _classify_document_content(self, file_path: str) -> Optional[str]:
        """Classify document content based on filename.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document type classification or None if not determined
        """
        # For real estate documents, the filename might provide hints
        filename = Path(file_path).name.lower()
        
        # Check filename for document type hints
        if re.search(r'mls|property|real.?estate|appraisal|tax|zoning', filename):
            return "Real Estate Document"
            
        return None
