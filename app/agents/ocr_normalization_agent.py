import os
import re
import json
import csv
import pytesseract
from PIL import Image
import pdfplumber
import fitz  # PyMuPDF
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from pathlib import Path

# Import our custom utilities
from ..utils.document_type_detector import DocumentTypeDetector
from ..utils.ocr_preprocessor import OCRPreprocessor


class OCRNormalizationAgent:
    """Agent responsible for extracting and normalizing text from documents.
    
    This agent handles various document types (PDF, images) and applies OCR
    when necessary to extract text, followed by cleaning and normalization.
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None, ocr_config: Optional[Dict[str, Any]] = None):
        """Initialize the OCR & Normalization Agent.
        
        Args:
            tesseract_cmd: Optional path to tesseract executable
            ocr_config: Optional configuration for OCR processing
        """
        # OCR configuration
        self.ocr_config = ocr_config or {}
        self.tesseract_config = self.ocr_config.get('tesseract_config', '--psm 6')  # Default to single block mode
        
        # Configure tesseract path - try multiple common locations for Windows
        if tesseract_cmd:
            # If explicitly provided, use it
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            # First try to get path from config file
            config_path = self.ocr_config.get('tesseract_path')
            if config_path and os.path.exists(config_path):
                pytesseract.pytesseract.tesseract_cmd = config_path
                logger.info(f"Found Tesseract from config at: {config_path}")
            else:
                # Try common Windows locations
                possible_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                    r'C:\Tesseract-OCR\tesseract.exe'
                ]
                
                for path in possible_paths:
                    if path and os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        logger.info(f"Found Tesseract at: {path}")
                        break
                
                # If all else fails, explicitly set to the most common path
                if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
                    default_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                    if os.path.exists(default_path):
                        pytesseract.pytesseract.tesseract_cmd = default_path
                        logger.info(f"Setting Tesseract to default path: {default_path}")
        
        # Initialize document type detector and OCR preprocessor
        self.document_detector = DocumentTypeDetector()
        self.ocr_preprocessor = OCRPreprocessor()
        
        # Check if tesseract is available
        self.tesseract_available = self._check_tesseract()
        
        # Path to store OCR logs
        self.log_dir = self.ocr_config.get('log_dir', './ocr_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Fields to track for each document
        self.required_fields = [
            'parcel_id', 'tax_value', 'property_address', 'flood_zone',
            'amount', 'bedrooms', 'bathrooms', 'year_built', 'mls_number'
        ]
        
        # Path patterns to ignore in documents (headers, footers, etc.)
        self.patterns_to_ignore = [
            r'Page \d+ of \d+',  # Page numbers
            r'^\s*\d+\s*$',  # Standalone page numbers
            r'^\s*-\s*\d+\s*-\s*$',  # Another page number format
            r'(?i)confidential',  # Confidentiality notices
            r'(?i)all rights reserved',  # Copyright notices
        ]
        
        if self.tesseract_available:
            logger.info("OCR & Normalization Agent initialized with Tesseract OCR support")
        else:
            logger.warning("OCR & Normalization Agent initialized WITHOUT Tesseract OCR support - text extraction from images will be limited")
    
    def _check_tesseract(self) -> bool:
        """Check if tesseract is available.
        
        Returns:
            bool: True if tesseract is available, False otherwise
        """
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            logger.warning("Please install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")
            return False

    def _analyze_field_coverage(self, text: str) -> Tuple[Dict[str, str], List[str]]:
        """Analyze the extracted text to determine which fields were extracted and which are missing.
        
        Args:
            text: The extracted text from the document
            
        Returns:
            Tuple of (extracted_fields, missing_fields)
        """
        extracted_fields = {}
        missing_fields = []
        
        # Check for each required field in the text
        for field in self.required_fields:
            # Replace underscores with spaces for pattern matching
            field_pattern = field.replace('_', ' ')
            
            # Common patterns for field labels
            patterns = [
                rf'(?i){field_pattern}\s*[:#]?\s*(\S[^\n\r]*)',  # Field: Value
                rf'(?i){field_pattern}\s*[=]\s*(\S[^\n\r]*)',   # Field = Value
                rf'(?i)"?{field_pattern}"?\s*[:\-]\s*(\S[^\n\r]*)' # "Field": Value
            ]
            
            # Special patterns for specific fields
            if field == 'amount' or field == 'tax_value':
                # Match currency amounts like $1,234.56 or 1,234.56
                patterns.extend([
                    r'(?i)(?:amount|value|price|cost|list price|listing price|sale price)\s*[:#]?\s*[$]?([\d,]+\.?\d*)',
                    r'(?i)[$]([\d,]+\.?\d*)'
                ])
            elif field == 'property_address':
                # Match address patterns
                patterns.extend([
                    r'(?i)(?:property|location|address)\s*[:#]?\s*(\d+\s+[^\n\r]+)',
                    r'(?i)(\d+\s+[A-Za-z0-9\s]+(?:Road|Street|Avenue|Lane|Drive|Circle|Blvd|Boulevard|Rd|St|Ave|Ln|Dr)[^\n\r]*)'
                ])
            elif field == 'parcel_id':
                # Match parcel ID patterns
                patterns.extend([
                    r'(?i)(?:parcel|tax|apn|pin)\s*(?:id|number|#)?\s*[:#]?\s*(\S[^\n\r]*)',
                    r'(?i)(?:parcel|tax|apn|pin)[#:\s]*(\d[\d\-]*)'
                ])
            elif field == 'mls_number':
                # Match MLS number patterns
                patterns.extend([
                    r'(?i)(?:mls|listing)\s*(?:number|no|#)\s*[:#]?\s*(\w+_?\d+)',
                    r'(?i)(?:mls|listing)[#:\s]*(\w+_?\d+)',
                    r'(?i)mls[_\s]*(\d+)'  # Simple MLS followed by digits
                ])
            elif field == 'date':
                # Match date patterns
                patterns.extend([
                    r'(?i)(?:date|listing date|sale date)\s*[:#]?\s*(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})',
                    r'(?i)(?:date|listing date|sale date)\s*[:#]?\s*([A-Za-z]+\s+\d{1,2},?\s*\d{4})'
                ])
            
            # Try each pattern to find the field
            field_found = False
            for pattern in patterns:
                matches = re.search(pattern, text)
                if matches and matches.group(1).strip():
                    extracted_value = matches.group(1).strip()
                    # Clean up the value
                    extracted_value = re.sub(r'\s+', ' ', extracted_value)
                    
                    # Special handling for currency values
                    if field in ['amount', 'tax_value'] and '$' not in extracted_value:
                        extracted_value = '$' + extracted_value
                    
                    extracted_fields[field] = extracted_value
                    field_found = True
                    break
            
            if not field_found:
                missing_fields.append(field)
        
        # Special check for MLS Number in the text - this is a common field that may be formatted differently
        if 'mls_number' in missing_fields:
            # Look for MLS followed by digits anywhere in the text
            mls_match = re.search(r'(?i)\bMLS[_\s]*(?:Number)?[_\s:#]*([A-Za-z0-9_]+)\b', text)
            if mls_match:
                extracted_fields['mls_number'] = mls_match.group(1).strip()
                missing_fields.remove('mls_number')
        
        return extracted_fields, missing_fields

    def _log_field_coverage(self, file_path: str, extracted_fields: Dict[str, str], missing_fields: List[str], ocr_type: str) -> None:
        """Log field coverage to a CSV file.
        
        Args:
            file_path: Path to the processed document
            extracted_fields: Dictionary of extracted fields
            missing_fields: List of missing fields
            ocr_type: Type of document processing used
        """
        file_name = os.path.basename(file_path)
        log_file = os.path.join(self.log_dir, 'field_coverage.csv')
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'file_name', 'extracted_fields', 'missing_fields', 'ocr_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'file_name': file_name,
                'extracted_fields': ','.join(extracted_fields.keys()),
                'missing_fields': ','.join(missing_fields),
                'ocr_type': ocr_type
            })
        
        # Also log as JSON for more detailed information
        json_log_file = os.path.join(self.log_dir, 'field_coverage_detailed.json')
        
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_name': file_name,
            'extracted_fields': extracted_fields,
            'missing_fields': missing_fields,
            'ocr_type': ocr_type
        }
        
        # Append to JSON file
        try:
            if os.path.isfile(json_log_file):
                with open(json_log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
                
            logs.append(log_entry)
            
            with open(json_log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error writing JSON log: {e}")

    
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document and extract normalized text.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict containing extracted text and processing metadata
        """
        try:
            # First, detect document type (digital or scanned)
            document_info = self.document_detector.detect_document_type(file_path)
            document_type = document_info['document_type']
            needs_ocr = document_info['needs_ocr']
            rotation_angle = document_info['rotation_angle']
            
            logger.info(f"Detected document type: {document_type}, needs OCR: {needs_ocr}, rotation: {rotation_angle}°")
            
            # Extract text based on document type
            if document_type == 'digital_pdf':
                # Use text extraction for digital PDFs
                text, metadata = await self._process_digital_pdf(file_path)
            elif document_type == 'scanned_pdf':
                # Use OCR for scanned PDFs
                text, metadata = await self._process_scanned_pdf(file_path, rotation_angle)
            elif document_type == 'scanned_image':
                # Use OCR for images
                text, metadata = await self._process_image(file_path, rotation_angle)
            else:
                # Fallback to file extension-based processing
                file_extension = Path(file_path).suffix.lower()
                if file_extension in [".pdf"]:
                    text, metadata = await self._process_pdf(file_path)
                elif file_extension in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                    text, metadata = await self._process_image(file_path)
                elif file_extension in [".csv", ".txt"]:
                    text, metadata = await self._process_text_file(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_extension}")
                    raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Normalize the extracted text
            normalized_text = self._normalize_text(text)
            
            # Extract and log field coverage
            extracted_fields, missing_fields = self._analyze_field_coverage(normalized_text)
            
            # Log field coverage
            self._log_field_coverage(file_path, extracted_fields, missing_fields, document_type)
            
            # Prepare the response
            result = {
                "extracted_text": normalized_text,
                "original_file": os.path.basename(file_path),
                "extraction_method": metadata.get("method", "unknown"),
                "page_count": metadata.get("page_count", 1),
                "confidence": metadata.get("confidence", None),
                "document_type": document_type,
                "ocr_type": "digital" if not needs_ocr else "scanned",
                "extracted_fields": extracted_fields,
                "missing_fields": missing_fields
            }
            
            logger.info(f"Successfully extracted text from {os.path.basename(file_path)}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    async def _process_digital_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process a digital PDF document with embedded text.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            text = ""
            page_count = 0
            
            # Extract text using PyMuPDF (fitz) for better performance
            doc = fitz.open(file_path)
            page_count = len(doc)
            
            for page_num in range(page_count):
                page = doc[page_num]
                page_text = page.get_text("text") or ""
                text += page_text + "\n\n"
            
            doc.close()
            
            # Prepare metadata
            metadata = {
                "method": "digital_pdf_extraction",
                "page_count": page_count,
                "confidence": 95  # High confidence for digital PDFs
            }
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error processing digital PDF {file_path}: {str(e)}")
            raise
    
    async def _process_scanned_pdf(self, file_path: str, rotation_angle: int = 0) -> Tuple[str, Dict[str, Any]]:
        """Process a scanned PDF document using OCR.
        
        Args:
            file_path: Path to the PDF file
            rotation_angle: Detected rotation angle to correct
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            if not self.tesseract_available:
                logger.warning("Tesseract not available for OCR processing")
                return "[OCR unavailable - Tesseract not installed]", {"method": "none", "page_count": 0, "confidence": 0}
            
            text = ""
            
            try:
                # Use pdf2image to convert PDF pages to images
                from pdf2image import convert_from_path
                
                # Convert PDF to images
                images = convert_from_path(file_path, dpi=300)
                page_count = len(images)
                text_content = []
                confidence_sum = 0
                
                # Process first 5 pages maximum to avoid long processing times
                pages_to_process = min(page_count, 5)
                
                for i in range(pages_to_process):
                    # Save page as temporary image
                    temp_img_path = os.path.join(tempfile.gettempdir(), f"page_{i}.png")
                    images[i].save(temp_img_path)
                    
                    # Preprocess the image if needed
                    if rotation_angle != 0:
                        preprocessed_path = self.ocr_preprocessor.preprocess_image(temp_img_path, rotation_angle)
                    else:
                        preprocessed_path = temp_img_path
                    
                    # Apply OCR with advanced configuration
                    ocr_config = f"--psm 6 -l eng {self.tesseract_config}"
                    page_text = pytesseract.image_to_string(preprocessed_path, config=ocr_config)
                    
                    # Get confidence data
                    ocr_data = pytesseract.image_to_data(preprocessed_path, output_type=pytesseract.Output.DICT, config=ocr_config)
                    page_confidence = 0
                    if 'conf' in ocr_data and len(ocr_data['conf']) > 0:
                        # Filter out -1 confidence values (which indicate no confidence data)
                        conf_values = [c for c in ocr_data['conf'] if c != -1]
                        if conf_values:
                            page_confidence = sum(conf_values) / len(conf_values)
                            confidence_sum += page_confidence
                    
                    # Append the text
                    text_content.append(page_text)
                    
                    # Clean up temporary files
                    try:
                        if os.path.exists(temp_img_path):
                            os.remove(temp_img_path)
                        if preprocessed_path != temp_img_path and os.path.exists(preprocessed_path):
                            os.remove(preprocessed_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file: {e}")
                
                # Combine text from all pages
                text = "\n\n".join(text_content)
                
                # Calculate average confidence
                avg_confidence = confidence_sum / pages_to_process if pages_to_process > 0 else 0
                
                # Prepare metadata
                metadata = {
                    "method": "ocr_scanned_pdf",
                    "page_count": page_count,
                    "confidence": round(avg_confidence, 2),
                    "rotation_corrected": rotation_angle != 0
                }
                
                return text, metadata
                
            except ImportError as e:
                logger.warning(f"pdf2image not available, falling back to basic extraction: {e}")
                # Fallback to simpler extraction
                return await self._process_pdf(file_path)
                
        except Exception as e:
            logger.error(f"Error processing scanned PDF {file_path}: {str(e)}")
            raise
    
    async def _process_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from a PDF file.
        
        This method uses pdfplumber to extract text from PDFs.
        If insufficient text is found, it falls back to OCR using an image-based approach.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted text, metadata)
        """
        # Try using pdfplumber for text extraction
        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                text_content = []
                
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text_content.append(page_text)
                
                text = "\n\n".join(text_content)
                
                # If we got reasonable text content, return it
                if len(text.strip()) > 100:  # Basic heuristic for "enough" text
                    return text, {"method": "pdfplumber_text_extraction", "page_count": page_count}
        
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")
        
        # If pdfplumber failed or didn't extract enough text, try OCR if possible
        logger.info(f"PDF text extraction limited or failed, trying alternative methods")
        
        # Try using pdf2image if available
        try:
            # Check if pdf2image is available
            # First attempt: pdf2image if available
            try:
                from pdf2image import convert_from_path
                logger.info("Using pdf2image for PDF to image conversion")
                
                # Convert PDF to images
                images = convert_from_path(file_path, dpi=300)
                page_count = len(images)
                text_content = []
                total_confidence = 0.0
                
                # Process each image with OCR
                for i, img in enumerate(images):
                    # Apply OCR if available
                    if self.tesseract_available:
                        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                        page_text = " ".join([word for word in ocr_data["text"] if word.strip()])
                        text_content.append(page_text)
                    else:
                        text_content.append("[OCR unavailable - Tesseract not installed]")
                    
                    # Calculate average confidence for this page
                    if self.tesseract_available and ocr_data["conf"] and len(ocr_data["conf"]) > 0:
                        valid_conf = [c for c in ocr_data["conf"] if c != -1]  # Filter out -1 values
                        if valid_conf:
                            total_confidence += sum(valid_conf) / len(valid_conf)
                
                text = "\n\n".join(text_content)
                avg_confidence = round(total_confidence / page_count, 2) if page_count > 0 and self.tesseract_available else None
                
                return text, {"method": "pdf_ocr", "page_count": page_count, "confidence": avg_confidence}
                
            except ImportError:
                # Fallback: Just use the text we already extracted with pdfplumber
                logger.info("pdf2image not available, using pdfplumber for basic extraction only")
                
                # Re-open with pdfplumber as a fallback
                with pdfplumber.open(file_path) as pdf:
                    page_count = len(pdf.pages)
                    text_content = []
                    
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            text_content.append(page_text)
                        else:
                            text_content.append("[Could not extract text from this page - OCR required]")
                    
                    text = "\n\n".join(text_content)
                    
                    return text, {"method": "pdfplumber_text_extraction_fallback", "page_count": page_count}
            
            for i, img in enumerate(images):
                # Apply OCR if available
                if self.tesseract_available:
                    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                    page_text = " ".join([word for word in ocr_data["text"] if word.strip()])
                    text_content.append(page_text)
                else:
                    text_content.append("[OCR unavailable - Tesseract not installed]")
                
                # Calculate average confidence for this page
                if self.tesseract_available and ocr_data["conf"] and len(ocr_data["conf"]) > 0:
                    valid_conf = [c for c in ocr_data["conf"] if c != -1]  # Filter out -1 values
                    if valid_conf:
                        total_confidence += sum(valid_conf) / len(valid_conf)
            
            text = "\n\n".join(text_content)
            avg_confidence = round(total_confidence / page_count, 2) if page_count > 0 and self.tesseract_available else None
            
            return text, {"method": "pdf_ocr", "page_count": page_count, "confidence": avg_confidence}
            
        except ImportError:
            logger.error("pdf2image not installed. For OCR of PDF files, please install: pip install pdf2image poppler-utils")
    
    async def _process_image(self, file_path: str, rotation_angle: int = 0) -> Tuple[str, Dict[str, Any]]:
        """Extract text from an image file using OCR with preprocessing.
        
        Args:
            file_path: Path to the image file
            rotation_angle: Detected rotation angle to correct
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            if not self.tesseract_available:
                logger.warning("Tesseract not available for OCR processing")
                return "[OCR unavailable - Tesseract not installed]", {"method": "none", "page_count": 1, "confidence": 0}
            
            # Preprocess the image for better OCR results
            preprocessed_path = self.ocr_preprocessor.preprocess_image(file_path, rotation_angle)
            
            # Apply OCR with advanced configuration
            ocr_config = f"--psm 6 -l eng {self.tesseract_config}"
            
            # Get both text and data for confidence calculation
            text = pytesseract.image_to_string(preprocessed_path, config=ocr_config)
            ocr_data = pytesseract.image_to_data(preprocessed_path, output_type=pytesseract.Output.DICT, config=ocr_config)
            
            # Calculate confidence
            confidence = 0
            if 'conf' in ocr_data and len(ocr_data['conf']) > 0:
                # Filter out -1 confidence values (which indicate no confidence data)
                conf_values = [c for c in ocr_data['conf'] if c != -1]
                if conf_values:
                    confidence = sum(conf_values) / len(conf_values)
            
            # Clean up temporary file if it's not the original
            try:
                if os.path.exists(preprocessed_path) and preprocessed_path != file_path and "_preprocessed" in preprocessed_path:
                    os.remove(preprocessed_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {preprocessed_path}: {e}")
                
            # Prepare metadata
            metadata = {
                "method": "image_ocr",
                "page_count": 1,  # Images are single page
                "confidence": round(confidence, 2),
                "rotation_corrected": rotation_angle != 0
            }
            
            # Apply OCR if available
            if self.tesseract_available:
                ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                
                # Extract text and confidence
                text = " ".join([word for word in ocr_data["text"] if word.strip()])
            else:
                text = "[OCR unavailable - Tesseract not installed]"
                ocr_data = {"conf": []}
            
            # Calculate average confidence
            confidence = None
            if ocr_data["conf"] and len(ocr_data["conf"]) > 0:
                valid_conf = [c for c in ocr_data["conf"] if c != -1]  # Filter out -1 values
                if valid_conf:
                    confidence = round(sum(valid_conf) / len(valid_conf), 2)
            
            return text, {"method": "image_ocr", "page_count": 1, "confidence": confidence}
            
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {str(e)}")
            raise
    
    def _normalize_text(self, text: str) -> str:
        """Normalize extracted text to improve field extraction.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
            
        # Remove patterns to ignore (headers, footers, etc.)
        for pattern in self.patterns_to_ignore:
            text = re.sub(pattern, "", text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks (keep paragraph structure)
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple newlines with exactly two
        
        # Fix OCR errors in numbers and currency
        # Convert currency formatting issues (e.g., $6/500.00 to $6,500.00)
        text = re.sub(r'\$(\d+)/(\d+)', r'$\1,\2', text)
        # Fix other slash-based number errors
        text = re.sub(r'(\d+)/(\d+)', r'\1,\2', text)
        
        # Normalize common OCR errors
        replacements = {
            # Common OCR mistakes
            r'l\d': 'ld',  # Fix common OCR error with '1' vs 'l'
            r'\$l': '$1',  # Dollar amounts with 'l' instead of '1'
            r'\sl': 's1',  # Numbers with 'l' instead of '1'
            r'\bO\b': '0',  # Standalone 'O' is often a '0'
            
            # Common field format corrections
            r'(?i)parcel\s*#': 'parcel_id:',  # Standardize parcel ID label
            r'(?i)tax\s*id': 'parcel_id:',     # Standardize tax ID label
            r'(?i)purchase\s*price': 'amount:', # Standardize price label
            r'(?i)sale\s*price': 'amount:',     # Standardize price label
            r'(?i)property\s*value': 'tax_value:', # Standardize value label
            r'(?i)MLS\s*Number\s*[:#]?\s*(\w+)': 'mls_number: \1', # Extract MLS number
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text.strip()
    
    async def _process_text_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from a plain text file (CSV, TXT).
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            return text, {"method": "text_file_read", "page_count": 1}
            
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
            
            return text, {"method": "text_file_read_latin1", "page_count": 1}
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            raise
    
    def _normalize_text(self, text: str) -> str:
        """Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Normalized text
        """
        # Replace multiple whitespace with single space
        normalized = re.sub(r'\s+', ' ', text)
        
        # Remove common headers and footers
        for pattern in self.patterns_to_ignore:
            normalized = re.sub(pattern, '', normalized)
        
        # Split by lines, remove empty ones, and rejoin
        lines = normalized.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        normalized = '\n'.join(non_empty_lines)
        
        # Replace multiple newlines with double newline
        normalized = re.sub(r'\n{3,}', '\n\n', normalized)
        
        return normalized.strip()
    
    def redact_pii(self, text: str) -> str:
        """Optional: Redact personally identifiable information (PII).
        
        Args:
            text: Normalized text
            
        Returns:
            Text with PII redacted
        """
        # Redact email addresses
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL REDACTED]', text)
        
        # Redact phone numbers (various formats)
        text = re.sub(r'\(\d{3}\)\s*\d{3}[-.\s]\d{4}', '[PHONE REDACTED]', text)  # (123) 456-7890
        text = re.sub(r'\d{3}[-.\s]\d{3}[-.\s]\d{4}', '[PHONE REDACTED]', text)  # 123-456-7890
        
        # Redact SSNs
        text = re.sub(r'\d{3}[-.\s]\d{2}[-.\s]\d{4}', '[SSN REDACTED]', text)
        
        return text
