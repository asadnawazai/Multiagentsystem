import os
import sys
import pytesseract
from PIL import Image
import numpy as np
import cv2
from loguru import logger

def test_tesseract_installation():
    """Test if Tesseract OCR is properly installed and configured."""
    print("\n" + "="*50)
    print("Tesseract OCR Installation Test")
    print("="*50)
    
    # Step 1: Check if pytesseract can find the Tesseract executable
    try:
        # Explicitly set the path to the Tesseract executable
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"\nTesseract path: {tesseract_path}")
        
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
    except Exception as e:
        print(f"\n❌ Error accessing Tesseract: {e}")
        print("\nPlease check if Tesseract is installed and the path is correctly configured.")
        return False
    
    # Step 2: Create a simple test image with text
    try:
        img = np.ones((200, 600), dtype=np.uint8) * 255
        cv2.putText(img, "Testing Tesseract OCR", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save the test image
        test_img_path = "tesseract_test.png"
        cv2.imwrite(test_img_path, img)
        
        print(f"\n✅ Created test image: {test_img_path}")
    except Exception as e:
        print(f"\n❌ Error creating test image: {e}")
        return False
    
    # Step 3: Run OCR on the test image
    try:
        text = pytesseract.image_to_string(Image.open(test_img_path))
        print(f"\n✅ OCR test successful!")
        print(f"   Extracted text: '{text.strip()}'")
        
        # Clean up
        os.remove(test_img_path)
        print(f"✅ Removed test image")
        
        return True
    except Exception as e:
        print(f"\n❌ OCR test failed: {e}")
        print("\nPlease ensure Tesseract is properly installed and configured.")
        return False

def test_project_integration():
    """Test if Tesseract is correctly integrated with the project configuration."""
    print("\n" + "-"*50)
    print("Project Integration Test")
    print("-"*50)
    
    try:
        # Import the OCRNormalizationAgent from the project
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from app.agents.ocr_normalization_agent import OCRNormalizationAgent
        
        # Initialize the agent with default configuration
        agent = OCRNormalizationAgent()
        
        print(f"\n✅ Successfully imported OCRNormalizationAgent")
        print(f"   Tesseract available: {agent.tesseract_available}")
        
        return agent.tesseract_available
    except Exception as e:
        print(f"\n❌ Error testing project integration: {e}")
        return False

if __name__ == "__main__":
    # Test Tesseract installation
    installation_ok = test_tesseract_installation()
    
    if installation_ok:
        # Test project integration
        integration_ok = test_project_integration()
        
        if integration_ok:
            print("\n✅ All tests passed! Tesseract OCR is properly installed and configured.")
            print("\nYou can now run your application and it should be able to process scanned documents.")
        else:
            print("\n❌ Project integration test failed.")
            print("\nPlease check your project configuration and ensure the OCR agent can find Tesseract.")
    else:
        print("\n❌ Tesseract installation test failed.")
        print("\nPlease ensure Tesseract is properly installed before proceeding.")
