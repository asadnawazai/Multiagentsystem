import os
import cv2
import numpy as np
import tempfile
from typing import Tuple, Dict, Any, Optional
from loguru import logger

class OCRPreprocessor:
    """Utility for preprocessing images before OCR to improve text recognition accuracy."""
    
    def __init__(self):
        """Initialize the OCR preprocessor."""
        pass
    
    def preprocess_image(self, image_path: str, rotation_angle: int = 0) -> str:
        """Preprocess an image for OCR processing.
        
        Args:
            image_path: Path to the image file
            rotation_angle: Known rotation angle to correct (0, 90, 180, or 270)
            
        Returns:
            Path to the preprocessed image
        """
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return image_path
            
            # Step 1: Correct rotation if needed
            if rotation_angle != 0:
                img = self._rotate_image(img, rotation_angle)
            
            # Step 2: Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Step 3: Apply adaptive thresholding for better contrast
            # This helps with text recognition in varying lighting conditions
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Step 4: Noise removal (optional)
            kernel = np.ones((1, 1), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Step 5: Deskew text (correct slight rotations)
            deskewed = self._deskew(binary)
            
            # Save the preprocessed image to a temporary file
            filename, ext = os.path.splitext(os.path.basename(image_path))
            output_path = os.path.join(tempfile.gettempdir(), f"{filename}_preprocessed{ext}")
            cv2.imwrite(output_path, deskewed)
            
            logger.info(f"Image preprocessed successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image_path  # Return original path if preprocessing fails
    
    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate an image by a specified angle.
        
        Args:
            image: OpenCV image array
            angle: Rotation angle in degrees (90, 180, or 270)
            
        Returns:
            Rotated image
        """
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270 or angle == -90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            return image
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Deskew an image to straighten text lines.
        
        Args:
            image: Binary image array
            
        Returns:
            Deskewed image
        """
        try:
            # Calculate skew angle
            coords = np.column_stack(np.where(image > 0))
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Limit deskewing to reasonable angles to avoid excessive rotation
            if abs(angle) > 10:
                return image
            
            # Rotate to correct skew
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
        except Exception as e:
            logger.error(f"Error deskewing image: {e}")
            return image  # Return original image if deskewing fails
    
    def preprocess_pdf_page(self, pdf_path: str, page_num: int = 0) -> str:
        """Extract and preprocess a page from a PDF for OCR.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to extract (0-based index)
            
        Returns:
            Path to the preprocessed image
        """
        try:
            import fitz  # PyMuPDF
            
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            if page_num >= len(doc):
                page_num = 0  # Default to first page if out of range
                
            # Get the page
            page = doc[page_num]
            
            # Convert to image
            pix = page.get_pixmap(dpi=300)  # Higher DPI for better OCR
            
            # Save as temporary image
            filename = os.path.splitext(os.path.basename(pdf_path))[0]
            temp_img_path = os.path.join(tempfile.gettempdir(), f"{filename}_page{page_num}.png")
            pix.save(temp_img_path)
            
            doc.close()
            
            # Preprocess the extracted image
            return self.preprocess_image(temp_img_path)
            
        except Exception as e:
            logger.error(f"Error preprocessing PDF page: {e}")
            return ""  # Return empty string if processing fails
