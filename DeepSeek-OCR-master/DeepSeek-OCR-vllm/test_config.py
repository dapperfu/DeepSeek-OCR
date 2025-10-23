# Test configuration for demonstrating image annotation
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6
MAX_CONCURRENCY = 1  # Reduced for testing
NUM_WORKERS = 1       # Reduced for testing
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'

# Test paths - using example image from assets
INPUT_PATH = '../assets/show1.jpg'
OUTPUT_PATH = 'test_output'

# Test prompt for document conversion
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# Import tokenizer for testing
from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)