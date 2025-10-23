# Test configuration for image annotation
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6
MAX_CONCURRENCY = 1
NUM_WORKERS = 1
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'

# Test paths
INPUT_PATH = 'assets/show1.jpg'
OUTPUT_PATH = 'annotation_test_output'

# Grounding prompt for layout detection and annotation
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
