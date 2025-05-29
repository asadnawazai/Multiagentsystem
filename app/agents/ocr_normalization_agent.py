import os
import cv2
import pytesseract
import numpy as np
import tempfile
import re
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
from PIL import Image
import fitz  # PyMuPDF
from ..utils.ocr_preprocessor import OCRPreprocessor
from ..utils.document_type_detector import DocumentTypeDetector
from ..utils.extract_fields_util import extract_full_document_text, TextExtractionEnhancer


class OCRNormalizationAgent:
    """Agent for OCR text extraction and normalization.
    
    This agent handles OCR processing for different document types,
    with specialized processing for digital PDFs, scanned PDFs, and images.
    It also handles text normalization to standardize extracted content.
    """
    
    def __init__(self, ocr_config: Optional[Dict[str, Any]] = None):
        """Initialize OCR & Normalization Agent.
        
        Args:
            ocr_config: Configuration options for OCR processing
        """
        # Default configuration
        self.config = {
            'tesseract_path': None,
            'tesseract_config': '--psm 6 -l eng',  # Page segmentation mode 6, English language
            'confidence_threshold': 20,  # Minimum confidence score to accept OCR results
            'rotation_correction': True,  # Whether to attempt to correct rotation
            'enable_preprocessing': True,  # Whether to preprocess images before OCR
            'log_dir': None  # Directory to save OCR logs and debug images
        }
        
        # Override defaults with provided config
        if ocr_config:
            self.config.update(ocr_config)
            
        # Set up Tesseract path
        if self.config['tesseract_path']:
            pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_path']
        else:
            # Try to find Tesseract in common locations
            common_locations = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                '/usr/bin/tesseract',
                '/usr/local/bin/tesseract'
            ]
            
            for path in common_locations:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info(f"Found Tesseract at: {path}")
                    break
        
        # Initialize OCR preprocessor
        self.preprocessor = OCRPreprocessor()
        
        # Initialize document type detector
        self.document_detector = DocumentTypeDetector()
        
        # Set up log directory if specified
        if self.config['log_dir']:
            os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Initialize temp directory for processing
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize PDF tools conditionally
        try:
            import fitz  # PyMuPDF
            import pdfplumber
            self.pdf_tools_available = True
        except ImportError:
            logger.warning("PDF processing libraries not available. PDF support will be limited.")
            self.pdf_tools_available = False
            
        # Check if Tesseract is installed and working
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"OCR & Normalization Agent initialized with Tesseract OCR support")
        except Exception as e:
            logger.warning(f"Could not initialize Tesseract OCR: {e}")
            logger.warning("OCR functionality will be limited")
    
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document with appropriate OCR based on document type.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict containing extracted text and processing metadata
        """
        try:
            logger.info(f"Starting enhanced extraction process for {file_path}")
            
            # Use our enhanced extraction utility for maximum accuracy
            extraction_result = extract_full_document_text(
                file_path, 
                tesseract_path=self.config.get('tesseract_path')
            )
            
            # Get the conventional document type info for metadata
            document_info = self.document_detector.detect_document_type(file_path)
            document_type = document_info['document_type']
            needs_ocr = document_info['needs_ocr']
            rotation_angle = document_info.get('rotation_angle', 0)
            
            logger.info(f"Processing document: {document_type}, OCR Needed: {needs_ocr}")
            
            # Extract text using our enhanced extraction utility
            extracted_text = extraction_result['extracted_text']
            extraction_method = extraction_result['extraction_method']
            extraction_confidence = extraction_result['confidence']
            
            # Apply additional normalization if needed
            normalized_text = self._normalize_text(extracted_text)
            
            # Detailed logging to help with client demo
            logger.info(f"Extraction complete: {len(normalized_text)} chars")
            logger.info(f"Method: {extraction_method}, Confidence: {extraction_confidence:.2f}")
            
            # Make sure to flag if extraction failed or produced minimal results
            if len(normalized_text) < 50:
                logger.warning("Extraction yielded minimal text! Check document or OCR settings.")
            
            # Create comprehensive result with metadata
            result = {
                "extracted_text": normalized_text,
                "text": normalized_text,  # For backward compatibility
                "document_type": document_type,
                "confidence": extraction_confidence,
                "extraction_method": extraction_method,
                "char_count": len(normalized_text),
                "page_count": extraction_result.get("page_count", 1),
                "rotation_corrected": rotation_angle != 0,
                "rotation_angle": rotation_angle,
                "needs_ocr": needs_ocr
            }
            
            logger.info(f"Successfully extracted text from {Path(file_path).name}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
    
    async def _process_digital_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from a digital PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        if not self.pdf_tools_available:
            logger.warning("PDF processing libraries not available, falling back to OCR")
            return await self._process_scanned_pdf(file_path, 0)
        
        try:
            import fitz  # PyMuPDF
            
            text = ""
            page_count = 0
            
            # Extract text using PyMuPDF
            doc = fitz.open(file_path)
            page_count = len(doc)
            
            for page in doc:
                text += page.get_text()
            
            doc.close()
            
            # If text extraction yields too little text, fallback to OCR
            if len(text.strip()) < 100:
                logger.info("Digital PDF text extraction yielded minimal text, falling back to OCR")
                return await self._process_scanned_pdf(file_path, 0)
            
            return text, {"page_count": page_count, "confidence": 100}
                
        except Exception as e:
            logger.error(f"Error extracting text from digital PDF: {str(e)}")
            # Fallback to OCR in case of error
            return await self._process_scanned_pdf(file_path, 0)
    
    async def _process_scanned_pdf(self, file_path: str, rotation_angle: int = 0) -> Tuple[str, Dict[str, Any]]:
        """Process a scanned PDF document using OCR.
        
        Args:
            file_path: Path to the PDF file
            rotation_angle: Detected rotation angle
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        if not self.pdf_tools_available:
            logger.warning("PDF processing libraries not available")
            return "PDF processing not available", {"confidence": 0, "page_count": 0}
        
        try:
            import fitz  # PyMuPDF
            
            # Initialize variables
            all_text = ""
            total_confidence = 0
            page_count = 0
            
            # Open PDF document
            doc = fitz.open(file_path)
            page_count = len(doc)
            
            # Process each page
            for page_num, page in enumerate(doc):
                # Save page as image
                pix = page.get_pixmap()
                img_path = os.path.join(self.temp_dir, f"page_{page_num}.png")
                pix.save(img_path)
                
                # Process the image with OCR
                text, metadata = await self._process_image(img_path, rotation_angle)
                
                # Add page number and append to result
                all_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
                
                # Add to confidence calculation
                total_confidence += metadata.get("confidence", 0)
                
                # Remove temporary image
                try:
                    os.remove(img_path)
                except Exception:
                    pass
            
            # Calculate average confidence
            avg_confidence = total_confidence / max(page_count, 1)
            
            return all_text, {"page_count": page_count, "confidence": avg_confidence}
            
        except Exception as e:
            logger.error(f"Error processing scanned PDF: {str(e)}")
            raise
    
    async def _process_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Generic PDF processing, which detects if OCR is needed.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        # First try direct text extraction
        text, metadata = await self._process_digital_pdf(file_path)
        
        # If text is very short, PDF might be scanned - try OCR
        if len(text.strip().split()) < 20:
            logger.info("PDF appears to be scanned or has limited text, using OCR")
            return await self._process_scanned_pdf(file_path, 0)
        
        return text, metadata
    
    async def _process_image(self, file_path: str, rotation_angle: int = 0) -> Tuple[str, Dict[str, Any]]:
        """Process an image file with OCR.
        
        Args:
            file_path: Path to the image file
            rotation_angle: Detected rotation angle
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            # Preprocess the image if enabled
            if self.config['enable_preprocessing']:
                # Use our OCR preprocessor to enhance image quality
                preprocessed_path = self.preprocessor.preprocess_image(file_path)
                if preprocessed_path:
                    logger.info(f"Image preprocessed successfully: {preprocessed_path}")
                    file_path = preprocessed_path
            
            # Load the image
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"Could not read image file: {file_path}")
            
            # Apply rotation correction if needed
            if rotation_angle != 0 and self.config['rotation_correction']:
                # Rotate the image to correct orientation
                rows, cols = image.shape[:2]
                rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_angle, 1)
                image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
                
            # Log the start of OCR processing
            logger.info(f"Starting OCR processing on {file_path}")
            
            # Convert to RGB for Tesseract
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use Tesseract OCR to extract text
            ocr_result = pytesseract.image_to_data(image, config=self.config['tesseract_config'], output_type=pytesseract.Output.DICT)
            
            # Combine text and calculate confidence
            text = ""
            confidence_sum = 0
            confidence_count = 0
            
            for i in range(len(ocr_result["text"])):
                word = ocr_result["text"][i]
                conf = ocr_result["conf"][i]
                
                # Skip empty text or low confidence results
                if not word.strip() or conf < self.config['confidence_threshold']:
                    continue
                    
                # Add space between words, but not after punctuation
                if text and not text.endswith((".", ",", "!", "?", ":", ";", "-")):
                    text += " "
                    
                text += word
                confidence_sum += conf
                confidence_count += 1
            
            # Calculate average confidence
            avg_confidence = confidence_sum / max(confidence_count, 1)  # Avoid division by zero
            
            logger.info(f"OCR completed successfully with confidence: {avg_confidence:.2f}")
            
            return text, {"confidence": avg_confidence, "page_count": 1}
        
        except Exception as e:
            logger.error(f"Error during OCR processing: {str(e)}")
            return "", {"confidence": 0, "page_count": 0}
    
    def _normalize_text(self, text: str) -> str:
        """Normalize extracted text for better processing.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
            
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove strange characters that might come from OCR errors
        # Use a simpler pattern that doesn't rely on Unicode properties
        text = re.sub(r'[^\w\s.,;:!?\-\(\)"\[\]]', '', text)
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.!?])([^\s])', r'\1 \2', text)
        
        # Replace repeated punctuation
        text = re.sub(r'([.!?])[.!?]+', r'\1', text)
        
        return text.strip()
