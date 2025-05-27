import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from loguru import logger

class DocumentTypeDetector:
    """Utility for detecting if a document is digital (with text layer) or scanned (requiring OCR)."""
    
    def __init__(self):
        """Initialize the document type detector."""
        pass
        
    def detect_document_type(self, file_path: str) -> Dict[str, Any]:
        """Detect if a document is digital or scanned.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict with document_type (digital_pdf, scanned_pdf, or scanned_image) and confidence
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        result = {
            "document_type": "unknown",
            "confidence": 0.0,
            "needs_ocr": True,
            "rotation_angle": 0
        }
        
        # Check if it's an image file
        if file_extension in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            result["document_type"] = "scanned_image"
            result["confidence"] = 1.0
            result["needs_ocr"] = True
            
            # Check for rotation
            rotation = self._detect_image_rotation(file_path)
            result["rotation_angle"] = rotation
            
            return result
        
        # Check if it's a PDF file
        if file_extension == ".pdf":
            return self._analyze_pdf(file_path)
        
        # Default to scanned_image for unknown file types
        return result
    
    def _analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze a PDF to determine if it's digital or scanned.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict with document_type and confidence
        """
        result = {
            "document_type": "unknown",
            "confidence": 0.0,
            "needs_ocr": True,
            "rotation_angle": 0
        }
        
        try:
            # Open the PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            total_pages = len(doc)
            pages_with_text = 0
            text_chars = 0
            
            for page_num in range(min(total_pages, 5)):  # Check first 5 pages max
                page = doc[page_num]
                text = page.get_text("text")
                
                if len(text.strip()) > 50:  # If page has substantial text
                    pages_with_text += 1
                    text_chars += len(text)
            
            # Calculate confidence based on text presence
            if total_pages > 0:
                text_confidence = pages_with_text / min(total_pages, 5)
            else:
                text_confidence = 0.0
            
            # Determine document type
            if text_confidence > 0.5:  # More than half of pages have text
                result["document_type"] = "digital_pdf"
                result["confidence"] = text_confidence
                result["needs_ocr"] = False
            else:
                result["document_type"] = "scanned_pdf"
                result["confidence"] = 1.0 - text_confidence
                result["needs_ocr"] = True
                
                # Check if we need to detect rotation for scanned PDFs
                if result["document_type"] == "scanned_pdf":
                    # Convert first page to image and check rotation
                    page = doc[0]
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    
                    # Save temporarily and check rotation
                    temp_path = os.path.join(os.path.dirname(pdf_path), "_temp_rotation_check.png")
                    with open(temp_path, "wb") as f:
                        f.write(img_data)
                    
                    rotation = self._detect_image_rotation(temp_path)
                    result["rotation_angle"] = rotation
                    
                    # Remove temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            doc.close()
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing PDF: {e}")
            result["document_type"] = "scanned_pdf"  # Default to scanned if analysis fails
            result["confidence"] = 0.5
            return result
    
    def _detect_image_rotation(self, image_path: str) -> int:
        """Detect if an image is rotated and needs correction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Rotation angle in degrees (0, 90, 180, or 270)
        """
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                return 0
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Basic method: check if height > width for potential rotation
            height, width = gray.shape
            if height > width * 1.25:  # If height is significantly greater than width
                # Further analyze with text line detection
                # Using horizontal lines in a rotated document means they're actually vertical
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
                
                if lines is not None:
                    # Count horizontal vs vertical lines
                    horizontal_lines = 0
                    vertical_lines = 0
                    
                    for line in lines:
                        rho, theta = line[0]
                        # Near horizontal lines
                        if (theta < 0.1 or abs(theta - np.pi) < 0.1 or abs(theta - 2*np.pi) < 0.1):
                            horizontal_lines += 1
                        # Near vertical lines
                        elif (abs(theta - np.pi/2) < 0.1 or abs(theta - 3*np.pi/2) < 0.1):
                            vertical_lines += 1
                    
                    # If more vertical than horizontal lines in a tall image, likely rotated 90°
                    if vertical_lines > horizontal_lines * 1.5:
                        return 90
            
            return 0  # No rotation detected
            
        except Exception as e:
            logger.error(f"Error detecting image rotation: {e}")
            return 0  # Default to no rotation
