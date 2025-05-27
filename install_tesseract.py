import os
import sys
import subprocess
import platform

def check_tesseract_installed():
    """Check if Tesseract OCR is installed and available."""
    try:
        # Try to import pytesseract
        import pytesseract
        print("\n✅ pytesseract is installed")
        
        # Check if Tesseract executable is found
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract OCR is installed (version {version})")
            print(f"   Path: {pytesseract.pytesseract.tesseract_cmd}")
            return True
        except Exception as e:
            print("❌ pytesseract is installed but Tesseract executable not found")
            print(f"   Error: {e}")
            return False
    except ImportError:
        print("❌ pytesseract is not installed")
        return False

def install_instructions():
    """Display instructions for installing Tesseract OCR."""
    system = platform.system()
    print("\n" + "-"*50)
    print("Tesseract OCR Installation Instructions:")
    print("-"*50)
    
    if system == "Windows":
        print("\nWindows Installation Steps:")
        print("1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Run the installer and follow the instructions")
        print("3. Add the Tesseract installation directory to your PATH")
        print("   - Default location: C:\\Program Files\\Tesseract-OCR\\")
        print("4. After installation, update your code with:")
        print("   pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
    elif system == "Darwin":  # macOS
        print("\nmacOS Installation Steps:")
        print("1. Install using Homebrew: brew install tesseract")
    elif system == "Linux":
        print("\nLinux (Ubuntu/Debian) Installation Steps:")
        print("1. Install using apt: sudo apt install tesseract-ocr")
        print("2. Install language data: sudo apt install tesseract-ocr-eng")
    
    print("\nAfter installation, run this script again to verify.")
    print("-"*50)

def configure_tesseract():
    """Configure Tesseract in the application."""
    import importlib
    import configparser
    
    try:
        # Check common locations
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Tesseract-OCR\tesseract.exe'
        ]
        
        tesseract_path = None
        for path in possible_paths:
            if os.path.exists(path):
                tesseract_path = path
                break
        
        if not tesseract_path:
            print("\n❓ Please enter the full path to tesseract.exe:")
            tesseract_path = input("> ").strip('"')
        
        # Test the provided path
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        version = pytesseract.get_tesseract_version()
        print(f"\n✅ Successfully configured Tesseract v{version} at {tesseract_path}")
        
        # Update configuration
        config_path = os.path.join('app', 'config', 'app_config.ini')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        config = configparser.ConfigParser()
        if os.path.exists(config_path):
            config.read(config_path)
        
        if 'OCR' not in config:
            config['OCR'] = {}
        
        config['OCR']['tesseract_path'] = tesseract_path
        
        with open(config_path, 'w') as f:
            config.write(f)
        
        print(f"✅ Updated configuration in {config_path}")
        return True
    except Exception as e:
        print(f"❌ Error configuring Tesseract: {e}")
        return False

def test_ocr():
    """Test OCR on a sample image."""
    try:
        from PIL import Image
        import pytesseract
        import numpy as np
        import cv2
        
        # Create a simple test image with text
        img = np.ones((200, 600), dtype=np.uint8) * 255
        cv2.putText(img, "Testing Tesseract OCR", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save the test image
        test_img_path = "tesseract_test.png"
        cv2.imwrite(test_img_path, img)
        
        # Run OCR
        text = pytesseract.image_to_string(Image.open(test_img_path))
        
        # Clean up
        os.remove(test_img_path)
        
        print(f"\n✅ OCR Test successful")
        print(f"Extracted text: '{text.strip()}'")
        return True
    except Exception as e:
        print(f"❌ OCR Test failed: {e}")
        return False

def main():
    print("\n" + "="*50)
    print("Tesseract OCR Configuration Helper")
    print("="*50)
    
    installed = check_tesseract_installed()
    
    if not installed:
        install_instructions()
        return
    
    # If installed, ask to configure and test
    print("\nWould you like to configure Tesseract for the application? (y/n)")
    if input("> ").lower() in ('y', 'yes'):
        if configure_tesseract():
            print("\nWould you like to run a test to verify OCR is working? (y/n)")
            if input("> ").lower() in ('y', 'yes'):
                test_ocr()

if __name__ == "__main__":
    main()
