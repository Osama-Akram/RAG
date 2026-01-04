# RAG System

A production-ready Retrieval-Augmented Generation (RAG) system that answers questions from your documents using AI.

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/RAG.git
cd rag_system

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your Groq API key to .env

# Ingest documents
python main.py --ingest ./data

# Ask a question
python main.py --query "What is machine learning?"

# Or run interactive mode
python main.py --interactive
```

## 🛠️ Tech Stack

- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS
- **LLM**: Groq API (Llama 3.1 8B)
- **Document Processing**: PyPDF2, pypdf

## 📁 Project Structure

```
rag_system/
├── ingestion.py      # Document loading and chunking
├── embedding.py      # Text embedding with sentence-transformers
├── vector_store.py   # FAISS vector storage
├── retrieval.py      # Similarity search
├── generation.py     # LLM integration
├── main.py           # CLI entry point
├── data/             # Documents (PDF/TXT)
├── output/           # Vector index storage
└── requirements.txt
```

## 🔄 Workflow

1. **Ingestion**: Load and chunk documents
2. **Embedding**: Convert text to vectors
3. **Storage**: Store in FAISS index
4. **Retrieval**: Find similar chunks for query
5. **Generation**: Generate answer with LLM

## 📝 License

MIT
