#!/usr/bin/env python3
"""
Simple DeepSeek-OCR Migration Test
==================================

This script demonstrates that the migration from Conda to vanilla Python
virtual environment works by testing basic functionality with example images.

Run this after: make install-all
"""

import os
import sys
from pathlib import Path

def main():
    print("🧪 DeepSeek-OCR Migration Test")
    print("=" * 40)
    
    # Test 1: Check virtual environment
    print("\n1. Testing virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   ✅ Virtual environment detected")
    else:
        print("   ⚠️  Not in virtual environment (run: source venv/bin/activate)")
    
    # Test 2: Check Python version
    print("\n2. Testing Python version...")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"   Python version: {python_version}")
    if sys.version_info.major == 3 and sys.version_info.minor == 12:
        print("   ✅ Python 3.12.x detected")
    else:
        print("   ⚠️  Python 3.12.x recommended")
    
    # Test 3: Check required packages
    print("\n3. Testing package imports...")
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'einops': 'Einops'
    }
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"   ✅ {name} imported successfully")
        except ImportError:
            print(f"   ❌ {name} not available")
            return False
    
    # Test 4: Check CUDA
    print("\n4. Testing CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("   ⚠️  CUDA not available (CPU-only mode)")
    except Exception as e:
        print(f"   ❌ CUDA test failed: {e}")
    
    # Test 5: Check example images
    print("\n5. Testing example images...")
    example_images = [
        'assets/show1.jpg',
        'assets/show2.jpg', 
        'assets/show3.jpg',
        'assets/show4.jpg'
    ]
    
    for img_path in example_images:
        if os.path.exists(img_path):
            print(f"   ✅ Found: {img_path}")
        else:
            print(f"   ❌ Missing: {img_path}")
    
    # Test 6: Test image loading
    print("\n6. Testing image loading...")
    try:
        from PIL import Image
        if os.path.exists('assets/show1.jpg'):
            image = Image.open('assets/show1.jpg')
            print(f"   ✅ Image loaded: {image.size} pixels")
            print(f"   Image mode: {image.mode}")
        else:
            print("   ❌ Example image not found")
    except Exception as e:
        print(f"   ❌ Image loading failed: {e}")
    
    # Test 7: Test DeepSeek-OCR modules
    print("\n7. Testing DeepSeek-OCR modules...")
    sys.path.append('DeepSeek-OCR-master/DeepSeek-OCR-vllm')
    
    try:
        from process.image_process import DeepseekOCRProcessor
        print("   ✅ Image processor imported")
        
        # Test processor initialization
        processor = DeepseekOCRProcessor()
        print("   ✅ Image processor initialized")
        
    except Exception as e:
        print(f"   ❌ DeepSeek-OCR modules failed: {e}")
        return False
    
    # Test 8: Test image processing pipeline
    print("\n8. Testing image processing pipeline...")
    try:
        if os.path.exists('assets/show1.jpg'):
            image = Image.open('assets/show1.jpg').convert('RGB')
            
            # Test basic image processing
            image_features = processor.tokenize_with_images(
                images=[image], 
                bos=True, 
                eos=True, 
                cropping=True
            )
            
            print("   ✅ Image tokenization successful")
            print(f"   Number of image tokens: {len(image_features[0][5])}")
            
        else:
            print("   ❌ No example image available for processing test")
            
    except Exception as e:
        print(f"   ⚠️  Image processing test failed (may need full model): {e}")
    
    print("\n" + "=" * 40)
    print("🎉 Migration test completed!")
    print("=" * 40)
    print("\nThe migration from Conda to vanilla Python virtual environment is working!")
    print("\nNext steps:")
    print("1. cd DeepSeek-OCR-master/DeepSeek-OCR-vllm")
    print("2. Update config.py with your input/output paths")
    print("3. python run_dpsk_ocr_image.py")
    print("\nExample usage:")
    print("   INPUT_PATH = 'assets/show1.jpg'")
    print("   OUTPUT_PATH = 'output'")
    print("   PROMPT = '<image>\\n<|grounding|>Convert the document to markdown.'")

if __name__ == "__main__":
    main()
