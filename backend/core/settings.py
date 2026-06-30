"""
Application settings using Pydantic Settings.
Loads configuration from environment variables and .env file.
"""

from pydantic_settings import BaseSettings
from typing import Optional, Literal
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "Enterprise Agentic RAG Platform"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "rag_platform"
    db_user: str = "rag_user"
    db_password: str = "changeme"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_default_model: str = "qwen3:4b"
    ollama_timeout: int = 300       # raised — qwen3:4b on CPU can take 2-3 min for a full answer
    ollama_num_ctx: int = 4096      # context window size passed to Ollama in every request
    
    # Hugging Face
    hf_home: str = "./models"
    hf_default_model: str = "Qwen/Qwen2.5-3B-Instruct"
    hf_cache_dir: Optional[str] = None
    
    # LLM Provider
    default_provider: Literal["ollama", "huggingface", "openai", "anthropic", "gemini", "azure"] = "ollama"
    fallback_provider: Optional[str] = "huggingface"
    
    # Generation defaults
    default_temperature: float = 0.7
    default_max_tokens: int = 512
    
    # RAG Configuration (Phase 1)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_separators: list[str] = ["\n\n", "\n", " ", ""]
    top_k_retrieval: int = 5
    
    # Document Processing (Phase 1)
    max_file_size: int = 10485760  # 10MB
    allowed_extensions: list[str] = [".pdf", ".docx"]
    upload_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    vectorstore_dir: str = "data/vectorstore"
    faiss_index_path: str = "data/vectorstore/faiss_index.bin"
    metadata_path: str = "data/vectorstore/metadata.json"
    
    # BM25 & Hybrid Retrieval (Phase 2)
    bm25_index_path: str = "data/vectorstore/bm25_index.pkl"
    bm25_k1: float = 1.5  # Term frequency saturation
    bm25_b: float = 0.75  # Length normalization
    default_retrieval_method: Literal["hybrid", "faiss", "bm25"] = "hybrid"
    faiss_weight: float = 0.5  # Weight for FAISS in hybrid retrieval
    bm25_weight: float = 0.5  # Weight for BM25 in hybrid retrieval
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    
    # Query Understanding (Phase 3)
    enable_query_reformulation: bool = True
    enable_query_expansion: bool = True
    enable_hyde: bool = True
    num_query_expansions: int = 3
    query_expansion_temperature: float = 0.7
    query_reformulation_temperature: float = 0.3
    hyde_temperature: float = 0.7
    hyde_max_tokens: int = 200          # keep short — hypothetical answer only needs 2-3 sentences
    query_expansion_max_tokens: int = 150   # 3 short questions fit in ~150 tokens
    query_reformulation_max_tokens: int = 80    # single rewritten question
    use_technical_hyde: bool = False

    # Reranking (Phase 4)
    enable_reranking: bool = False  # off by default — requires model download on first use
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 3         # return this many results after reranking
    reranker_batch_size: int = 16

    # Evaluation (Phase 5)
    eval_judge_model: str = ""      # blank = use ollama_default_model
    eval_judge_provider: str = ""   # blank = use default_provider
    eval_max_samples: int = 50      # cap per evaluation run

    # Conversational Memory (Phase 6)
    memory_session_ttl: int = 86400           # Redis TTL per conversation in seconds (24 h)
    memory_max_history_messages: int = 20     # messages injected into the prompt window
    memory_enable_summarisation: bool = True  # summarise old turns when history grows large
    memory_summary_threshold: int = 10        # summarise when turns exceed this value

    # Safety & Governance — Guardrails (Phase 7)
    guardrails_enable_injection: bool = True    # prompt injection / jailbreak detection
    guardrails_enable_toxicity:  bool = True    # toxic language detection
    guardrails_enable_pii:       bool = True    # PII detection & redaction
    guardrails_enable_hallucination: bool = True  # hallucination / grounding check on output
    guardrails_block_on_injection:   bool = True   # block request when injection detected
    guardrails_block_on_toxicity:    bool = True   # block request when toxic content detected
    guardrails_block_on_pii_input:   bool = False  # warn only by default (don't block on PII in input)
    guardrails_block_on_hallucination: bool = False  # warn only by default
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    log_max_size: int = 10485760  # 10MB
    log_backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS
    cors_origins: list[str] = ["http://localhost:8501", "http://localhost:3000"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL database URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Made with Bob
