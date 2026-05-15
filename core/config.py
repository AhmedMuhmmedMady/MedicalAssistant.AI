import os
from dotenv import load_dotenv

load_dotenv()

# External API Keys
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "AIzaSyC1LE3N84XeK9VIHdFxOp2NSnf4r6QEYfw")
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY", "pcsk_3Bxu6E_HjF5cNUBvb5aQJ3qYmBmcGtfinhJuc1Gd1Kj5oJcxdQR4FtJjjJHFcMvzwxtPow")

if not GEMINI_API_KEY:   raise RuntimeError("❌ GEMINI_API_KEY is not set.")
if not PINECONE_API_KEY: raise RuntimeError("❌ PINECONE_API_KEY is not set.")

# Index Config
INDEX_NAME         = os.getenv("PINECONE_INDEX", "medical-index-arabicdata")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")

# RAG & Embeddings
MIN_CONFIDENCE   = float(os.getenv("SCORE_THRESHOLD", "0.60"))
TOP_K            = int(os.getenv("TOP_K", "7"))
EMBED_MODEL      = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM        = 384
MAX_CONTEXT_MATCHES = 3

# Hybrid Scoring Weights
WEIGHT_EXACT          = 0.20
WEIGHT_CATEGORY       = 0.10
EXACT_MATCH_THRESHOLD = 0.92

# Constraints
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
MAX_IMAGE_MB     = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
MAX_IMAGE_BYTES  = MAX_IMAGE_MB * 1024 * 1024

# Retry & Timeout
MAX_RETRIES  = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY  = float(os.getenv("RETRY_DELAY", "1.5"))
EXTERNAL_CALL_TIMEOUT   = int(os.getenv("EXTERNAL_CALL_TIMEOUT", "25"))

# Concurrency
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "20"))
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW   = 60
