# Configuration file for RAG service

# Model configurations
DEFAULT_MODEL_PATH = "path/to/your/llama/model"  # Update this
DEFAULT_MODEL_NAME = "LLaMA-2-7B"  # Update this

# Embedding model
EMBEDDING_MODEL = "facebook/contriever-msmarco"

# Chunking parameters
CHUNK_SIZE = 2048
OVERLAP = 100

# Retrieval parameters
DEFAULT_TOP_K = 3

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Directories
DATA_DIR = "data"
INDEX_DIR = "indexes"

# Supported file types
SUPPORTED_EXCEL_EXTENSIONS = ['.xlsx', '.xls']
SUPPORTED_WORD_EXTENSIONS = ['.docx']