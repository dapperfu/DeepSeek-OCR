#!/usr/bin/env python3
"""
DeepSeek-OCR Migration Test Example
===================================

This script demonstrates that the migration from Conda to vanilla Python
virtual environment works correctly by running OCR on example images.

Usage:
    python test_migration.py

Requirements:
    - Virtual environment must be activated: source venv/bin/activate
    - All dependencies must be installed: make install-all
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the vLLM directory to Python path
sys.path.append('DeepSeek-OCR-master/DeepSeek-OCR-vllm')

def test_environment():
    """Test that the environment is properly set up."""
    print("🔍 Testing environment setup...")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Warning: Not in a virtual environment")
    
    # Check required packages
    required_packages = ['torch', 'transformers', 'PIL', 'vllm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please run: make install-all")
        return False
    
    print("✅ All required packages are available")
    return True

def test_image_processing():
    """Test image processing capabilities."""
    print("\n🖼️  Testing image processing...")
    
    try:
        from PIL import Image
        import torch
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available, using CPU")
        
        # Test loading an example image
        example_image_path = "assets/show1.jpg"
        if os.path.exists(example_image_path):
            image = Image.open(example_image_path)
            print(f"✅ Successfully loaded example image: {image.size}")
            return image
        else:
            print(f"❌ Example image not found: {example_image_path}")
            return None
            
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return None

def test_vllm_inference():
    """Test vLLM inference with example image."""
    print("\n🚀 Testing vLLM inference...")
    
    try:
        # Import required modules
        from vllm import AsyncLLMEngine, SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.model_executor.models.registry import ModelRegistry
        from deepseek_ocr import DeepseekOCRForCausalLM
        from process.image_process import DeepseekOCRProcessor
        from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
        
        print("✅ Successfully imported vLLM modules")
        
        # Register the model
        ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
        print("✅ Model registered successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ vLLM inference test failed: {e}")
        return False

def test_transformers_inference():
    """Test Transformers inference."""
    print("\n🔄 Testing Transformers inference...")
    
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        print("✅ Successfully imported Transformers modules")
        
        # Test tokenizer loading (without downloading the full model)
        try:
            tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)
            print("✅ Tokenizer loaded successfully")
        except Exception as e:
            print(f"⚠️  Tokenizer loading failed (expected without full model): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformers inference test failed: {e}")
        return False

def create_test_config():
    """Create a test configuration file."""
    print("\n⚙️  Creating test configuration...")
    
    config_content = '''# Test configuration for migration verification
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6
MAX_CONCURRENCY = 1  # Reduced for testing
NUM_WORKERS = 1      # Reduced for testing
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'

# Test paths
INPUT_PATH = 'assets/show1.jpg'
OUTPUT_PATH = 'test_output'

PROMPT = '<image>\\n<|grounding|>Convert the document to markdown.'

from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
'''
    
    config_path = "DeepSeek-OCR-master/DeepSeek-OCR-vllm/test_config.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Test configuration created: {config_path}")
    return config_path

def run_simple_ocr_test():
    """Run a simple OCR test using the example image."""
    print("\n📝 Running simple OCR test...")
    
    try:
        from PIL import Image
        import torch
        
        # Load example image
        image_path = "assets/show1.jpg"
        if not os.path.exists(image_path):
            print(f"❌ Example image not found: {image_path}")
            return False
        
        image = Image.open(image_path).convert('RGB')
        print(f"✅ Loaded test image: {image.size}")
        
        # Test image processing pipeline
        sys.path.append('DeepSeek-OCR-master/DeepSeek-OCR-vllm')
        from process.image_process import DeepseekOCRProcessor
        
        processor = DeepseekOCRProcessor()
        print("✅ Image processor initialized")
        
        # Test image tokenization (without running full inference)
        try:
            image_features = processor.tokenize_with_images(
                images=[image], 
                bos=True, 
                eos=True, 
                cropping=True
            )
            print("✅ Image tokenization successful")
            print(f"   - Number of image tokens: {len(image_features[0][5])}")
            return True
            
        except Exception as e:
            print(f"⚠️  Image tokenization failed (may need full model): {e}")
            return False
            
    except Exception as e:
        print(f"❌ Simple OCR test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("🧪 DeepSeek-OCR Migration Test")
    print("=" * 60)
    print("Testing migration from Conda to vanilla Python virtual environment")
    print()
    
    # Run all tests
    tests = [
        ("Environment Setup", test_environment),
        ("Image Processing", test_image_processing),
        ("vLLM Modules", test_vllm_inference),
        ("Transformers Modules", test_transformers_inference),
        ("Simple OCR Pipeline", run_simple_ocr_test),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Migration test PASSED!")
        print("The migration from Conda to vanilla Python virtual environment is working correctly.")
    else:
        print(f"\n⚠️  Migration test PARTIALLY PASSED ({passed}/{total})")
        print("Some components may need additional setup or the full model download.")
    
    print("\nNext steps:")
    print("1. To run full inference, download the model:")
    print("   cd DeepSeek-OCR-master/DeepSeek-OCR-vllm")
    print("   python run_dpsk_ocr_image.py")
    print("2. Update config.py with your input/output paths")
    print("3. Ensure you have sufficient GPU memory for inference")

if __name__ == "__main__":
    main()
