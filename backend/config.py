import os
import logging
from dotenv import load_dotenv
from typing import List

# load environment variables
load_dotenv()

class Config:
    """Application configuration class"""

    def __init__(self):
        #Server settings
        self.HOST = os.getenv("HOST", "localhost")
        self.PORT = int(os.getenv("PORT", "8000"))
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"

        # File upload settings
        self.UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
        self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "100")) * 1024 * 1024
        self.ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "png,jpg,jpeg").split(",")

        # Logging settings
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
        
        # QwenAPI settings 
        self.DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
        self.BASE_URL = os.getenv("BASE_URL")
        self.QWEN_MODEL=os.getenv("QWEN_MODEL")
        self.TEMPERATURE=os.getenv("QWEN_TEMPERATURE")
        self.QWEN_MAX_TOKENS=os.getenv("QWEN_MAX_TOKENS")
        
        # Azure OpenAI embedding API settings 
        self.EMBEDDING_DEPLOYMENT=os.getenv("EMBEDDING_DEPLOYMENT")
        self.EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL")
        self.EMBEDDING_API_VERSION=os.getenv("EMBEDDING_API_VERSION")
        self.EMBEDDING_AZURE_ENDPOINT=os.getenv("EMBEDDING_AZURE_ENDPOINT")
        self.EMBEDDING_API_KEY=os.getenv("EMBEDDING_API_KEY")

        #  CHat settings
        self.MAX_CONVERSATION_HISTORY=int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
        self.CHAT_SYSTEM_PROMPT=os.getenv("CHAT_SYSTEM_PROMPT",
            "You are a helpful medical assistant. Provide accurate, helpful information about health and medical topics. Always remind users to consult healthcare professionals for serious medical concerns."
        )

        # RAG settings and vector DB
        self.VECTOR_DB_PATH=os.getenv("VECTOR_DB_PATH", "data/qdrant_db")
        self.DOCS_DB_PATH=os.getenv("DOCS_DB_PATH", "data/docs_db")
        self.PARSED_DOCS_PATH=os.getenv("PARSED_DOCS_PATH", "data/parsed_docs")
        self.QDRANT_HOST=os.getenv("QDRANT_HOST")  # If set, use remote Qdrant instead of local path
        self.QDRANT_PORT=int(os.getenv("QDRANT_PORT", "6333"))
        self.QDRANT_USE_SSL=os.getenv("QDRANT_USE_SSL", "False").lower() == "true"
        self.QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
        self.CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", "512"))
        self.CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP", "50"))
        self.MAX_RETRIEVAL_DOCS=int(os.getenv("MAX_RETRIEVAL_DOCS", "5"))
        self.RAG_CONFIDENCE_THRESHOLD=float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.6"))
        
        # Hybrid retrieval settings
        self.ENABLE_HYBRID_RETRIEVAL=os.getenv("ENABLE_HYBRID_RETRIEVAL", "True").lower() == "true"
        self.HYBRID_ALPHA=float(os.getenv("HYBRID_ALPHA", "0.7"))  # Weight for dense vs BM25 (0.7 = 70% dense, 30% BM25)
        
        # Query expansion settings
        self.ENABLE_QUERY_EXPANSION=os.getenv("ENABLE_QUERY_EXPANSION", "True").lower() == "true"
        self.QUERY_EXPANSION_TEMPERATURE=float(os.getenv("QUERY_EXPANSION_TEMPERATURE", "0.3"))  # Lower temperature for consistent expansion
        self.QUERY_EXPANSION_MAX_TOKENS=int(os.getenv("QUERY_EXPANSION_MAX_TOKENS", "200"))  # Shorter response for expansion
        
        # Cross-encoder reranking settings
        self.ENABLE_CROSS_ENCODER_RERANKING=os.getenv("ENABLE_CROSS_ENCODER_RERANKING", "True").lower() == "true"
        self.CROSS_ENCODER_MODEL=os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")  # Default cross-encoder model
        self.RERANK_TOP_K=int(os.getenv("RERANK_TOP_K", "20"))  # Number of documents to rerank
        
        # Multi-Agent settings
        self.AGENT_DECISION_MODEL=os.getenv("AGENT_DECISION_MODEL")
        self.AGENT_DECISION_TEMPERATURE=float(os.getenv("AGENT_DECISION_TEMPERATURE", "0.1"))
        self.AGENT_CONFIDENCE_THRESHOLD=float(os.getenv("AGENT_CONFIDENCE_THRESHOLD", "0.8"))
        self.ENABLE_AGENT_ROUTING=os.getenv("ENABLE_AGENT_ROUTING", "True").lower() == "true"
        
        # Computer Vision settings
        self.MODELS_PATH=os.getenv("MODELS_PATH", "models")
        self.BRAIN_TUMOR_MODEL_PATH=os.getenv("BRAIN_TUMOR_MODEL_PATH", "models/brain_tumor_model.pth")
        self.CHEST_XRAY_MODEL_PATH=os.getenv("CHEST_XRAY_MODEL_PATH", "models/chest_xray_model.pth")
        self.SKIN_LESION_MODEL_PATH=os.getenv("SKIN_LESION_MODEL_PATH", "models/skin_lesion_model.pth")
        self.IMAGE_PREPROCESSING_SIZE=int(os.getenv("IMAGE_PREPROCESSING_SIZE", "224"))
        self.CONFIDENCE_THRESHOLD=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))


        # Voice Processing settings
        self.AZURE_SPEECH_KEY=os.getenv("AZURE_SPEECH_KEY")
        self.AZURE_SPEECH_REGION=os.getenv("AZURE_SPEECH_REGION", "westus")
        self.AZURE_TTS_VOICE_NAME=os.getenv("AZURE_TTS_VOICE_NAME", "en-US-JennyNeural")
        self.AZURE_STT_VOICE_NAME=os.getenv("AZURE_STT_VOICE_NAME", "en-US-JennyNeural")
        self.SPEECH_RECOGNITION_LANGUAGE=os.getenv("SPEECH_RECOGNITION_LANGUAGE", "en-US")
        self.AUDIO_FORMAT=os.getenv("AUDIO_FORMAT", "wav")
        self.AUDIO_SAMPLE_RATE=int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
        
        # Web Search settings
        self.GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
        self.GOOGLE_CSE_ID=os.getenv("GOOGLE_CSE_ID")
        self.SEARCH_TIMEOUT=int(os.getenv("SEARCH_TIMEOUT", "10"))
        self.SEARCH_RESULTS_LIMIT=int(os.getenv("SEARCH_RESULTS_LIMIT", "5"))
        
        # Security and Validation settings
        self.RATE_LIMIT_PER_MINUTE=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        self.MAX_MESSAGE_LENGTH=int(os.getenv("MAX_MESSAGE_LENGTH", "2000"))
        # MAX_FILE_SIZE already set in upload settings (bytes); do not override here
        self.ENABLE_PROFANITY_FILTER=os.getenv("ENABLE_PROFANITY_FILTER", "True").lower() == "true"
        self.ENABLE_CONTENT_VALIDATION=os.getenv("ENABLE_CONTENT_VALIDATION", "True").lower() == "true"
        self.ENABLE_RATE_LIMITING=os.getenv("ENABLE_RATE_LIMITING", "True").lower() == "true"
        
        # Windows Service settings
        self.ENABLE_WINDOWS_SERVICE=os.getenv("ENABLE_WINDOWS_SERVICE", "False").lower() == "true"

        # Create necessary directories
        self._create_directories()
    
        # Setup logging
        self._setup_logging()

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.UPLOAD_FOLDER,
            os.path.join(self.UPLOAD_FOLDER, "frontend"),
            os.path.join(self.UPLOAD_FOLDER, "backend"),
            os.path.join(self.UPLOAD_FOLDER, "speech"),
            self.DOCS_DB_PATH,
            self.PARSED_DOCS_PATH,
            self.MODELS_PATH,
            "logs"
        ]
        # Only create local vector DB path when using embedded Qdrant
        if not self.QDRANT_HOST and self.VECTOR_DB_PATH:
            directories.append(self.VECTOR_DB_PATH)
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _setup_logging(self): 
        """Setup logging configuration""" 
        # Create handlers with UTF-8 encoding
        file_handler = logging.FileHandler(self.LOG_FILE, encoding='utf-8')
        console_handler = logging.StreamHandler()
        
        # Set formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL.upper()),
            handlers=[file_handler, console_handler]
        )

    def is_allowed_file(self, filename: str) -> bool:
        """Check if the file extension is allowed"""
        if '.' not in filename:
            return False
        extension = filename.rsplit('.', 1)[1].lower()
        return extension in self.ALLOWED_EXTENSIONS

    def get_max_file_size_mb(self) -> float:
        """Get the maximum file size in MB"""
        return self.MAX_FILE_SIZE / 1024 / 1024
    
    def get_max_file_size(self) -> float:
        """Get the maximum file size in MB (alias for compatibility)"""
        return self.get_max_file_size_mb()
    
    def __repr__(self):
        return f"Config(HOST={self.HOST}, PORT={self.PORT}, DEBUG={self.DEBUG}, UPLOAD_FOLDER={self.UPLOAD_FOLDER}, MAX_FILE_SIZE={self.get_max_file_size()} MB, ALLOWED_EXTENSIONS={self.ALLOWED_EXTENSIONS}, LOG_LEVEL={self.LOG_LEVEL}, LOG_FILE={self.LOG_FILE})"
    
# Global config instance
config = Config()

# Logger instance
logger = logging.getLogger(__name__)