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
            
            # Save the preprocessed image to a temporary file with absolute path
            # Use the uploads directory instead of system temp dir for better reliability
            upload_dir = os.path.dirname(os.path.abspath(image_path))
            filename, ext = os.path.splitext(os.path.basename(image_path))
            output_path = os.path.join(upload_dir, f"{filename}_preprocessed{ext}")
            
            # Ensure the image is properly written
            success = cv2.imwrite(output_path, deskewed)
            if not success:
                logger.error(f"Failed to write preprocessed image to {output_path}")
                return image_path  # Fallback to original if we can't save preprocessed
            
            # Verify the file exists before returning
            if os.path.exists(output_path):
                logger.info(f"Image preprocessed successfully: {output_path}")
                return output_path
            else:
                logger.error(f"Preprocessed file not found at {output_path} after writing")
                return image_path  # Fallback to original
            
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
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing steps to image for better OCR results.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        
        # First check if this appears to be a form document
        if self._is_likely_form(image):
            return self.preprocess_form_document(image)
            
        # If not a form, apply standard preprocessing
        # Step 1: Convert to grayscale if it's not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Step 4: Noise removal (optional)
            kernel = np.ones((1, 1), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Step 5: Deskew text (correct slight rotations)
            deskewed = self._deskew(binary)
            
            return deskewed
    
    def _is_likely_form(self, image: np.ndarray) -> bool:
        """Detect if image is likely a structured form document.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            True if image appears to be a form document
        """
        # Convert to grayscale if it's not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Look for horizontal and vertical lines which are common in forms
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return False
            
        # Count horizontal and vertical lines
        horizontal_lines = 0
        vertical_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            
            # Horizontal lines have angles close to 0 or 180
            if angle < 5 or angle > 175:
                horizontal_lines += 1
            # Vertical lines have angles close to 90
            elif 85 < angle < 95:
                vertical_lines += 1
                
        # If we have several horizontal and vertical lines, it's likely a form
        return horizontal_lines >= 5 and vertical_lines >= 5
        
    def preprocess_form_image(self, image: np.ndarray) -> np.ndarray:
        """Apply special preprocessing for form documents.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image optimized for form OCR
        """
        try:
            # Make a copy to avoid modifying the original
            img = image.copy()
            
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Apply contrast stretching to enhance features
            min_val, max_val = np.percentile(gray, (2, 98))
            stretched = np.clip((gray - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)

            # First sharpen the image to make text and checkboxes clearer
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(stretched, -1, kernel)
            
            # Apply adaptive thresholding with optimized parameters for forms
            binary = cv2.adaptiveThreshold(
                sharpened, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11,  # Smaller block size for forms with small text and checkboxes
                8     # C constant adjusted for better form field detection
            )
            
            # Use morphological operations to preserve checkboxes and form structures
            kernel = np.ones((2, 2), np.uint8)
            # Dilate to make checkboxes and lines more prominent
            dilated = cv2.dilate(binary, kernel, iterations=1)
            # Erode to clean up noise while preserving structure
            processed = cv2.erode(dilated, kernel, iterations=1)
            
            # Run Canny edge detection to emphasize checkbox boundaries
            edges = cv2.Canny(processed, 100, 200)
            kernel = np.ones((2, 2), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Combine processed image with edge detection to get both text and form structures
            result = cv2.bitwise_and(processed, cv2.bitwise_not(dilated_edges))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in form preprocessing: {str(e)}")
            return image  # Return original image on error
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
