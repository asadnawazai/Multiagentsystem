import os
import sys
import configparser
from PIL import Image
import cv2
import numpy as np
import pytesseract
from pathlib import Path
from loguru import logger

# Set up logging
logger.add("app/logs/ocr_test.log", rotation="500 KB")

def setup_tesseract():
    """Configure Tesseract based on config file."""
    config_path = os.path.join('app', 'config', 'ocr_config.ini')
    
    if os.path.exists(config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        
        if 'OCR' in config and 'tesseract_path' in config['OCR']:
            tesseract_path = config['OCR']['tesseract_path']
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info(f"Configured Tesseract path: {tesseract_path}")
                return True
    
    # Fallback to common locations
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"Found Tesseract at: {path}")
            return True
    
    logger.error("Tesseract executable not found")
    return False

def create_test_image():
    """Create a test image with sample real estate data."""
    # Create a white image
    img = np.ones((800, 600), dtype=np.uint8) * 255
    
    # Add some sample real estate text
    lines = [
        "REAL ESTATE LISTING",
        "Property Address: 123 Main Street, Anytown, CA 90210",
        "Parcel ID: 123-45-678",
        "List Price: $450,000",
        "Year Built: 1985",
        "Bedrooms: 3",
        "Bathrooms: 2.5",
        "Square Feet: 2,100",
        "Flood Zone: X",
        "MLS Number: MLS12345"
    ]
    
    # Add text to image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 0, 0)  # Black
    thickness = 2
    y = 50
    
    for i, line in enumerate(lines):
        # Make the first line bold (title)
        if i == 0:
            cv2.putText(img, line, (50, y), font, 1.0, color, 3)
            y += 60
        else:
            cv2.putText(img, line, (50, y), font, font_scale, color, thickness)
            y += 40
    
    # Save the image
    test_img_path = "real_estate_test.png"
    cv2.imwrite(test_img_path, img)
    logger.info(f"Created test image: {test_img_path}")
    
    return test_img_path

def test_ocr_extraction():
    """Test OCR extraction on a sample image."""
    if not setup_tesseract():
        print("❌ Tesseract not properly configured. OCR test failed.")
        return False
    
    try:
        # Create test image
        test_img_path = create_test_image()
        
        # Run OCR
        print("\nRunning OCR on test image...")
        text = pytesseract.image_to_string(Image.open(test_img_path))
        
        # Get OCR data for confidence
        ocr_data = pytesseract.image_to_data(Image.open(test_img_path), output_type=pytesseract.Output.DICT)
        
        # Calculate average confidence
        conf_values = [c for c in ocr_data['conf'] if c != -1]
        avg_confidence = sum(conf_values) / len(conf_values) if conf_values else 0
        
        # Display results
        print("\n" + "-"*50)
        print("OCR Test Results:")
        print("-"*50)
        print(f"\nExtracted Text:\n{text}")
        print(f"\nOCR Confidence: {avg_confidence:.2f}%")
        
        # Check if key fields were extracted
        fields_to_check = [
            "Property Address", "Parcel ID", "Price", "Year Built", 
            "Bedrooms", "Bathrooms", "Flood Zone", "MLS"
        ]
        
        print("\nField Detection:")
        for field in fields_to_check:
            if field.lower() in text.lower():
                print(f"✅ {field} detected")
            else:
                print(f"❌ {field} NOT detected")
        
        # Clean up
        os.remove(test_img_path)
        
        print("\n" + "-"*50)
        if avg_confidence > 80:
            print("✅ OCR Test PASSED - Good confidence level")
            return True
        else:
            print("⚠️ OCR Test PARTIAL - Low confidence level")
            return False
            
    except Exception as e:
        logger.error(f"Error during OCR test: {e}")
        print(f"\n❌ OCR Test FAILED: {e}")
        return False

def main():
    print("\n" + "="*50)
    print("PanoramaScore OCR System Test")
    print("="*50)
    
    test_ocr_extraction()

if __name__ == "__main__":
    main()
