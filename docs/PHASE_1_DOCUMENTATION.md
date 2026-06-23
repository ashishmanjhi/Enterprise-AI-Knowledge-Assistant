# Phase 1: Basic RAG - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Components](#components)
5. [API Reference](#api-reference)
6. [User Guide](#user-guide)
7. [Configuration](#configuration)
8. [Deployment](#deployment)

---

## Overview

Phase 1 implements a production-ready Retrieval-Augmented Generation (RAG) system for enterprise knowledge management. The system enables users to upload documents (PDF, DOCX), automatically processes and indexes them, and provides intelligent question-answering capabilities using local LLMs.

### Key Capabilities

- **Document Ingestion:** Upload and process PDF and DOCX files
- **Intelligent Chunking:** Automatic text segmentation with overlap
- **Semantic Search:** Vector-based similarity search using FAISS
- **RAG Chat:** Context-aware question answering with source attribution
- **Local LLM:** Uses Ollama for privacy and cost efficiency
- **Web Interface:** User-friendly Streamlit UI for document management and chat

### Technology Stack

- **Backend:** FastAPI, Python 3.9+
- **LLM:** Ollama (qwen3:4b default)
- **Embeddings:** BAAI/bge-small-en-v1.5 (384 dimensions)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Frontend:** Streamlit
- **Document Processing:** PyPDF2, python-docx
- **Text Splitting:** LangChain RecursiveCharacterTextSplitter

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Dashboard   │  │  Documents   │  │     Chat     │      │
│  │   (Home)     │  │  Management  │  │  Interface   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                    Streamlit UI                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/REST
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                         API Layer                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              FastAPI Application                      │   │
│  │  ┌────────────────┐      ┌────────────────┐         │   │
│  │  │ Document Routes│      │  Chat Routes   │         │   │
│  │  │  (6 endpoints) │      │ (4 endpoints)  │         │   │
│  │  └────────────────┘      └────────────────┘         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Business Logic Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Ingestion   │  │  Retrieval   │  │  RAG Chain   │      │
│  │   Pipeline   │  │   Service    │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Infrastructure Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Document   │  │  Embedding   │  │     LLM      │      │
│  │   Loaders    │  │   Service    │  │   Service    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │    FAISS     │  │   Metadata   │                        │
│  │ Vector Store │  │   Manager    │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       Storage Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Documents   │  │ Vector Index │  │   Metadata   │      │
│  │  (PDF/DOCX)  │  │    (FAISS)   │  │    (JSON)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      External Services                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Ollama (Local LLM Server)               │   │
│  │                 http://localhost:11434                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

#### Document Ingestion Flow

```
Upload → Validate → Load → Chunk → Embed → Index → Store
   │         │        │       │       │       │       │
   │         │        │       │       │       │       └─→ Metadata JSON
   │         │        │       │       │       └─→ FAISS Index
   │         │        │       │       └─→ Generate Embeddings
   │         │        │       └─→ Split into Chunks (1000 chars)
   │         │        └─→ Extract Text (PDF/DOCX)
   │         └─→ Check File Type & Size
   └─→ Receive File Upload
```

#### RAG Query Flow

```
Question → Embed → Search → Retrieve → Generate → Response
    │        │       │         │          │          │
    │        │       │         │          │          └─→ Answer + Sources
    │        │       │         │          └─→ LLM Generation
    │        │       │         └─→ Top-K Chunks
    │        │       └─→ FAISS Similarity Search
    │        └─→ Generate Query Embedding
    └─→ User Question
```

---

## Features

### 1. Document Management

#### Supported Formats

- **PDF:** Text-based PDFs (not image-based)
- **DOCX:** Microsoft Word documents

#### Document Processing

- **Automatic Text Extraction:** Extracts text from uploaded documents
- **Metadata Preservation:** Maintains filename, file type, size, upload date
- **Chunking Strategy:**
  - Chunk size: 1000 characters
  - Overlap: 200 characters
  - Preserves context across chunks
- **Page Tracking:** Maintains page numbers for source attribution

#### Document Operations

- **Upload:** Single file upload with progress tracking
- **List:** View all uploaded documents with metadata
- **Search:** Semantic search across all documents
- **Delete:** Remove documents and associated data
- **Statistics:** View system-wide document statistics

### 2. Semantic Search

#### Vector Search

- **Embedding Model:** BAAI/bge-small-en-v1.5
  - Dimension: 384
  - Optimized for English text
  - Fast inference on CPU
- **Vector Store:** FAISS IndexFlatL2
  - Exact similarity search
  - L2 distance metric
  - Persistent storage

#### Search Features

- **Top-K Retrieval:** Configurable number of results (default: 3)
- **Similarity Scoring:** Distance-based relevance scores
- **Metadata Filtering:** Filter by document, date, etc.
- **Source Attribution:** Each result includes document reference

### 3. RAG Chat

#### Question Answering

- **Context-Aware:** Uses retrieved documents as context
- **Source Citations:** Includes references to source documents
- **Streaming Support:** Real-time response generation
- **Fallback Mode:** Can operate without RAG for general questions

#### LLM Integration

- **Primary Provider:** Ollama (local)
  - Default model: qwen3:4b
  - No API keys required
  - Privacy-preserving
- **Optional Provider:** OpenAI
  - Requires API key
  - Cloud-based
  - Higher quality responses

#### Chat Features

- **Message History:** Maintains conversation context
- **Adjustable Parameters:**
  - Number of sources (top_k)
  - RAG enable/disable
  - Temperature (creativity)
  - Max tokens (response length)

### 4. User Interface

#### Dashboard (Home)

- System status overview
- Quick statistics
- Getting started guide
- Phase completion status

#### Documents Page

- File upload interface with drag-and-drop
- Document library with cards
- Search and filter capabilities
- Delete functionality
- Statistics dashboard

#### Chat Page

- Interactive chat interface
- Message history display
- Source citations
- Parameter controls
- Clear conversation option

---

## Components

### Backend Components

#### 1. Document Loaders (`backend/ingestion/loaders/`)

**Base Loader** (`base.py`)

```python
class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders"""

    @abstractmethod
    def load(self, file_path: str) -> Document:
        """Load document and extract text"""
        pass

    def validate_file(self, file_path: str) -> bool:
        """Validate file exists and is readable"""
        pass
```

**PDF Loader** (`pdf_loader.py`)

- Uses PyPDF2 for text extraction
- Handles multi-page documents
- Preserves page numbers
- Error handling for corrupted PDFs

**DOCX Loader** (`docx_loader.py`)

- Uses python-docx library
- Extracts paragraphs and tables
- Maintains document structure
- Handles formatting elements

#### 2. Text Chunking (`backend/ingestion/chunking.py`)

```python
class DocumentChunker:
    """Splits documents into overlapping chunks"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split document into chunks with metadata"""
        pass
```

**Chunking Strategy:**

- Recursive splitting by separators
- Preserves semantic boundaries
- Maintains metadata per chunk
- Configurable size and overlap

#### 3. Embeddings (`backend/llm/embeddings.py`)

```python
class EmbeddingService:
    """Generate embeddings for text chunks"""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        pass
```

**Features:**

- Batch processing for efficiency
- Normalization of embeddings
- Caching for repeated queries
- GPU support (if available)

#### 4. Vector Store (`backend/retrievers/vector_store.py`)

```python
class FAISSVectorStore:
    """FAISS-based vector store for similarity search"""

    def __init__(self, dimension: int = 384):
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def add_vectors(self, vectors: np.ndarray, metadata: List[dict]):
        """Add vectors with metadata to index"""
        pass

    def search(self, query_vector: np.ndarray, top_k: int = 3):
        """Search for similar vectors"""
        pass

    def save(self, path: str):
        """Persist index to disk"""
        pass

    def load(self, path: str):
        """Load index from disk"""
        pass
```

**Features:**

- Exact similarity search
- Persistent storage
- Metadata management
- Efficient batch operations

#### 5. Ingestion Pipeline (`backend/ingestion/pipeline.py`)

```python
class IngestionPipeline:
    """Orchestrates document ingestion process"""

    def __init__(self):
        self.loader_factory = DocumentLoaderFactory()
        self.chunker = DocumentChunker()
        self.embedding_service = EmbeddingService()
        self.vector_store = FAISSVectorStore()
        self.metadata_manager = MetadataManager()

    async def ingest_document(self, file_path: str) -> DocumentMetadata:
        """Complete ingestion pipeline"""
        # 1. Load document
        # 2. Chunk text
        # 3. Generate embeddings
        # 4. Index vectors
        # 5. Store metadata
        pass
```

**Pipeline Steps:**

1. File validation
2. Text extraction
3. Chunking
4. Embedding generation
5. Vector indexing
6. Metadata storage

#### 6. RAG Chain (`backend/llm/rag_chain.py`)

```python
class RAGChain:
    """Combines retrieval and generation"""

    def __init__(self):
        self.retriever = DocumentRetriever()
        self.llm_service = LLMService()

    async def generate_answer(
        self,
        question: str,
        top_k: int = 3,
        **kwargs
    ) -> RAGResponse:
        """Generate answer using RAG"""
        # 1. Retrieve relevant chunks
        # 2. Format context
        # 3. Generate answer
        # 4. Include sources
        pass
```

**RAG Process:**

1. Query embedding
2. Similarity search
3. Context formatting
4. Prompt construction
5. LLM generation
6. Source attribution

#### 7. LLM Service (`backend/llm/llm_service.py`)

```python
class LLMService:
    """Multi-provider LLM service"""

    def __init__(self):
        self.default_provider = "ollama"
        self.ollama_client = OllamaClient()
        self.openai_client = OpenAIClient()  # Optional

    async def generate(
        self,
        prompt: str,
        provider: str = None,
        model: str = None,
        **kwargs
    ) -> str:
        """Generate text using specified provider"""
        pass

    async def stream_generate(self, prompt: str, **kwargs):
        """Stream generation for real-time responses"""
        pass
```

**Supported Providers:**

- **Ollama (Default):**
  - Local execution
  - No API keys
  - Privacy-preserving
  - Models: qwen3:4b, llama2, mistral, etc.
- **OpenAI (Optional):**
  - Cloud-based
  - Requires API key
  - Models: gpt-3.5-turbo, gpt-4, etc.

### Frontend Components

#### 1. Main App (`frontend/streamlit/app.py`)

**Features:**

- Navigation sidebar
- System status display
- Quick statistics
- Getting started guide

#### 2. Documents Page (`frontend/streamlit/pages/1_📄_Documents.py`)

**Features:**

- File uploader (PDF, DOCX)
- Upload progress tracking
- Document library with cards
- Delete functionality
- Statistics dashboard

**UI Elements:**

- File uploader with drag-and-drop
- Document cards with metadata
- Delete confirmation
- Success/error messages

#### 3. Chat Page (`frontend/streamlit/pages/2_💬_Chat.py`)

**Features:**

- Chat input
- Message history
- Source citations
- Parameter controls

**UI Elements:**

- Chat input box
- Message bubbles (user/assistant)
- Source expandable sections
- Sidebar controls (top_k, RAG toggle)

---

## API Reference

### Document Endpoints

#### 1. Upload Document

```http
POST /api/documents/upload
Content-Type: multipart/form-data

Parameters:
- file: File (PDF or DOCX)

Response: 200 OK
{
  "document_id": "uuid",
  "filename": "document.pdf",
  "file_type": "pdf",
  "file_size": 1024000,
  "num_chunks": 45,
  "upload_date": "2026-06-23T10:00:00Z",
  "status": "completed"
}
```

#### 2. List Documents

```http
GET /api/documents/

Response: 200 OK
[
  {
    "document_id": "uuid",
    "filename": "document.pdf",
    "file_type": "pdf",
    "file_size": 1024000,
    "num_chunks": 45,
    "upload_date": "2026-06-23T10:00:00Z",
    "status": "completed"
  }
]
```

#### 3. Get Document

```http
GET /api/documents/{document_id}

Response: 200 OK
{
  "document_id": "uuid",
  "filename": "document.pdf",
  "file_type": "pdf",
  "file_size": 1024000,
  "num_chunks": 45,
  "upload_date": "2026-06-23T10:00:00Z",
  "status": "completed",
  "chunks": [...]
}
```

#### 4. Delete Document

```http
DELETE /api/documents/{document_id}

Response: 200 OK
{
  "message": "Document deleted successfully",
  "document_id": "uuid"
}
```

#### 5. Search Documents

```http
POST /api/documents/search
Content-Type: application/json

{
  "query": "What is machine learning?",
  "top_k": 3,
  "filters": {}
}

Response: 200 OK
{
  "results": [
    {
      "chunk_text": "Machine learning is...",
      "document_id": "uuid",
      "filename": "ml_guide.pdf",
      "page_number": 5,
      "similarity_score": 0.85
    }
  ],
  "total_results": 3
}
```

#### 6. Get Statistics

```http
GET /api/documents/stats

Response: 200 OK
{
  "total_documents": 10,
  "total_chunks": 450,
  "total_size": 10240000,
  "file_types": {
    "pdf": 7,
    "docx": 3
  }
}
```

### Chat Endpoints

#### 1. Chat (RAG)

```http
POST /api/chat/
Content-Type: application/json

{
  "message": "What are the key features?",
  "top_k": 3,
  "use_rag": true,
  "temperature": 0.7,
  "max_tokens": 500
}

Response: 200 OK
{
  "answer": "The key features include...",
  "sources": [
    {
      "chunk_text": "Feature 1: ...",
      "document_id": "uuid",
      "filename": "features.pdf",
      "page_number": 2
    }
  ],
  "metadata": {
    "num_sources": 3,
    "model": "qwen3:4b",
    "provider": "ollama"
  }
}
```

#### 2. Stream Chat

```http
POST /api/chat/stream
Content-Type: application/json

{
  "message": "Explain this concept",
  "top_k": 3
}

Response: 200 OK (Server-Sent Events)
data: {"chunk": "The", "done": false}
data: {"chunk": " concept", "done": false}
data: {"chunk": " is...", "done": true}
```

#### 3. Direct Chat (No RAG)

```http
POST /api/chat/direct
Content-Type: application/json

{
  "message": "What is 2+2?",
  "model": "qwen3:4b",
  "provider": "ollama"
}

Response: 200 OK
{
  "answer": "2+2 equals 4.",
  "model": "qwen3:4b",
  "provider": "ollama"
}
```

#### 4. Health Check

```http
GET /api/chat/health

Response: 200 OK
{
  "status": "healthy",
  "ollama_available": true,
  "openai_available": false,
  "default_provider": "ollama"
}
```

---

## User Guide

### Getting Started

#### 1. Prerequisites

- Python 3.9 or higher
- Ollama installed and running
- 4GB+ RAM
- 2GB+ disk space

#### 2. Installation

```bash
# Clone repository
git clone <repository-url>
cd Enterprise-AI-Knowledge-Assistant

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install and start Ollama
# Download from https://ollama.ai
ollama serve

# Pull default model
ollama pull qwen3:4b
```

#### 3. Start Services

**Terminal 1 - Backend:**

```bash
cd backend
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**

```bash
cd frontend/streamlit
streamlit run app.py
```

#### 4. Access Application

- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Using the System

#### Uploading Documents

1. Navigate to "📄 Documents" page
2. Click "Browse files" or drag-and-drop
3. Select PDF or DOCX file
4. Wait for processing (progress bar shown)
5. Document appears in library

**Tips:**

- Upload multiple documents for better context
- Ensure PDFs are text-based (not scanned images)
- Larger documents take longer to process

#### Searching Documents

1. Go to Documents page
2. Use search functionality
3. View results with relevance scores
4. Click to view full document

#### Chatting with Documents

1. Navigate to "💬 Chat" page
2. Type your question
3. Adjust parameters if needed:
   - Number of sources (1-10)
   - Enable/disable RAG
4. Submit question
5. View answer with sources

**Tips:**

- Be specific in questions
- Reference document names for targeted queries
- Check sources for accuracy
- Adjust number of sources for more context

#### Managing Documents

**View Documents:**

- See all uploaded documents in library
- View metadata (size, chunks, date)

**Delete Documents:**

- Click delete button on document card
- Confirm deletion
- Document and associated data removed

**View Statistics:**

- Total documents
- Total chunks
- Storage used
- File type distribution

---

## Configuration

### Backend Configuration (`backend/core/settings.py`)

```python
class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "Enterprise RAG API"
    API_VERSION: str = "1.0.0"

    # Storage Paths
    DOCUMENTS_DIR: str = "data/documents"
    VECTOR_STORE_DIR: str = "data/vector_store"
    METADATA_DIR: str = "data/metadata"

    # RAG Configuration
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    DEFAULT_TOP_K: int = 3

    # LLM Configuration
    DEFAULT_LLM_PROVIDER: str = "ollama"
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL: str = "qwen3:4b"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_DEFAULT_MODEL: str = "gpt-3.5-turbo"

    # File Upload
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx"]
```

### Environment Variables

Create `.env` file in project root:

```env
# Optional: OpenAI API Key
OPENAI_API_KEY=sk-...

# Optional: Custom Ollama Host
OLLAMA_HOST=http://localhost:11434

# Optional: Custom Model
OLLAMA_DEFAULT_MODEL=qwen3:4b

# Optional: Storage Paths
DOCUMENTS_DIR=data/documents
VECTOR_STORE_DIR=data/vector_store
```

### Customization

#### Change Embedding Model

```python
# In settings.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Update based on model
```

#### Change Chunk Size

```python
# In settings.py
CHUNK_SIZE = 1500  # Larger chunks
CHUNK_OVERLAP = 300  # More overlap
```

#### Change LLM Model

```python
# In settings.py
OLLAMA_DEFAULT_MODEL = "llama2"  # Or mistral, codellama, etc.
```

#### Add Custom Document Loader

```python
# Create new loader in backend/ingestion/loaders/
class CustomLoader(BaseDocumentLoader):
    def load(self, file_path: str) -> Document:
        # Custom loading logic
        pass

# Register in loader factory
```

---

## Deployment

### Local Development

Already covered in Getting Started section.

### Production Deployment

#### Docker Deployment (Recommended)

**Dockerfile:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/streamlit/app.py --server.port 8501 --server.address 0.0.0.0"]
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama
    ports:
      - '11434:11434'
    volumes:
      - ollama_data:/root/.ollama

  rag-app:
    build: .
    ports:
      - '8000:8000'
      - '8501:8501'
    volumes:
      - ./data:/app/data
    environment:
      - OLLAMA_HOST=http://ollama:11434
    depends_on:
      - ollama

volumes:
  ollama_data:
```

**Deploy:**

```bash
docker-compose up -d
```

#### Cloud Deployment

**AWS EC2:**

1. Launch EC2 instance (t3.medium or larger)
2. Install Docker and Docker Compose
3. Clone repository
4. Run docker-compose up
5. Configure security groups (ports 8000, 8501, 11434)

**Azure VM:**

1. Create VM (Standard_D2s_v3 or larger)
2. Install Docker
3. Deploy using docker-compose
4. Configure NSG rules

**GCP Compute Engine:**

1. Create VM instance
2. Install dependencies
3. Deploy application
4. Configure firewall rules

### Performance Optimization

#### 1. Use GPU for Embeddings

```python
# In embeddings.py
self.model = SentenceTransformer(model_name, device='cuda')
```

#### 2. Enable FAISS GPU

```python
# Install faiss-gpu instead of faiss-cpu
pip install faiss-gpu
```

#### 3. Batch Processing

```python
# Process multiple documents in parallel
# Adjust batch size in settings
EMBEDDING_BATCH_SIZE = 32
```

#### 4. Caching

```python
# Enable response caching
# Add Redis for distributed caching
```

### Monitoring

#### Logging

```python
# Configure logging in settings.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

#### Metrics

- Document processing time
- Query response time
- Embedding generation time
- Vector search latency
- LLM generation time

#### Health Checks

- `/api/chat/health` - LLM service status
- `/api/documents/stats` - System statistics
- Ollama availability check

---

## Troubleshooting

See [PHASE_1_TESTING_GUIDE.md](PHASE_1_TESTING_GUIDE.md) for comprehensive troubleshooting guide.

---

## Next Steps

### Phase 2 Enhancements (Planned)

- Multi-modal support (images, tables)
- Advanced chunking strategies
- Query expansion and rewriting
- Conversation memory
- Multi-document reasoning
- Custom prompt templates
- User authentication
- Document versioning

---

**Last Updated:** 2026-06-23  
**Version:** 1.0.0  
**Status:** Production Ready
