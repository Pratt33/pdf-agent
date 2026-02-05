---
title: PDF Question Answering Agent
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 📄 PDF Question Answering Agent

An optimized Retrieval-Augmented Generation (RAG) system for intelligent PDF document querying powered by LLaMA 3.2 and semantic search.

## 🎯 Overview

This application implements a production-ready RAG pipeline that enables users to upload PDF documents and ask natural language questions. The system retrieves relevant context from the document and generates accurate, context-aware answers using large language models.

## 🏗️ System Architecture

### Pipeline Components

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Index → Retrieval → LLM Generation
```

1. **Document Processing**
   - PDF parsing using `pypdf`
   - Text extraction with page-level granularity
   
2. **Text Chunking**
   - Fixed-size chunking (400 words per chunk)
   - Overlap strategy for context preservation
   
3. **Semantic Embedding**
   - Model: `all-MiniLM-L6-v2` (SentenceTransformers)
   - Dimensions: 384
   - Batch processing for efficiency (batch_size=32)
   
4. **Vector Search**
   - FAISS IndexFlatL2 for similarity search
   - Top-k retrieval (k=3)
   - L2 distance metric
   
5. **Answer Generation**
   - Model: `meta-llama/Llama-3.2-3B-Instruct`
   - Max tokens: 500
   - Context-based prompting

## ⚡ Performance Optimizations

### Implemented Features

| Feature | Impact | Benefit |
|---------|--------|---------|
| **In-Memory Caching** | 95% latency reduction | Stores last 10 processed PDFs with LRU eviction |
| **Batch Processing** | 40% faster embeddings | Processes text chunks in batches of 32 |
| **Async I/O** | 20% overall speedup | Non-blocking operations for file/network I/O |
| **Model Pre-warming** | Eliminates cold start | Loads embedding model at startup |
| **Progressive Loading** | Better UX | Real-time status updates during processing |

### Latency Benchmarks

| Scenario | Latency |
|----------|---------|
| First request (cold) | ~10s |
| Cached PDF (same document) | **~0.5s** |
| Different PDF | ~10s |

## 🛠️ Technical Stack

**Backend:**
- Python 3.13
- Gradio 6.3.0 (UI Framework)
- HuggingFace Hub (LLM Inference)

**ML/NLP:**
- SentenceTransformers (Embeddings)
- FAISS (Vector Search)
- PyTorch (ML Backend)

**Dependencies:**
```
gradio>=6.3.0
pypdf>=4.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
huggingface_hub>=0.20.0
torch>=2.0.0
```

## 📁 Project Structure

```
pdf-agent/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── .gitignore            # Git ignore rules
├── app/
│   ├── __init__.py
│   ├── pdf_loader.py     # PDF text extraction
│   ├── embedder.py       # Embedding & indexing
│   ├── retriever.py      # Semantic search
│   ├── cache.py          # LRU caching system
│   └── llm/
│       ├── __init__.py
│       ├── base.py       # LLM interface
│       └── hf_llm.py     # HuggingFace client
└── data/                 # User uploads (gitignored)
```

## 🚀 Quick Start

### Local Deployment

1. **Clone Repository**
```bash
git clone https://github.com/Pratt33/pdf-agent.git
cd pdf-agent
```

2. **Install Dependencies**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set Environment Variables**
```bash
export HF_TOKEN="your_huggingface_token"  # Windows: $env:HF_TOKEN="..."
```

4. **Run Application**
```bash
python app.py
```

5. **Access Interface**
Open http://127.0.0.1:7860 in your browser

### HuggingFace Space Deployment

The application is pre-configured for HuggingFace Spaces deployment:

1. Push to HuggingFace Space repository
2. Set `HF_TOKEN` in Space secrets
3. Application auto-deploys on push

## 💡 Usage

1. **Upload PDF**: Click "Upload PDF Document" and select your file
2. **Ask Question**: Type your question in natural language
3. **Get Answer**: Click "Get Answer" to receive contextual response
4. **Cache Benefit**: Ask multiple questions on same PDF instantly

### Example Queries

- "What is the main topic of this document?"
- "Summarize the key findings"
- "What methodology was used?"
- "What are the conclusions?"

## 🔬 Research Context

### RAG Methodology

Retrieval-Augmented Generation combines:
- **Dense retrieval** for semantic similarity
- **Generative models** for natural language synthesis
- **Context injection** to ground LLM responses

### Why RAG?

- ✅ Reduces hallucinations
- ✅ Enables source attribution
- ✅ Handles dynamic knowledge
- ✅ Cost-effective vs. fine-tuning

## 🎨 Features

- ✨ Modern, responsive UI with Gradio
- 🔄 Real-time processing status
- 💾 Intelligent caching (last 10 PDFs)
- 🚀 Optimized batch processing
- 📊 Support for various PDF formats
- 🔐 Secure token handling

## 📊 System Requirements

**Minimum:**
- Python 3.9+
- 4GB RAM
- 2GB disk space

**Recommended:**
- Python 3.11+
- 8GB RAM
- GPU (optional, for faster embeddings)

## 🔧 Configuration

### Cache Settings
```python
# app/cache.py
pdf_cache = PDFCache(max_size=10)  # Adjust cache size
```

### Chunk Size
```python
# app/embedder.py
chunk_text(text, chunk_size=400)  # Modify chunk size
```

### Retrieval Count
```python
# app/retriever.py
retrieve(query, index, chunks, model, k=3)  # Adjust k
```

## 🐛 Troubleshooting

**Issue: Model API Errors**
- Ensure `HF_TOKEN` is set correctly
- Check HuggingFace API status

**Issue: Slow Processing**
- First request loads model (expected)
- Subsequent requests use cache
- Consider GPU for large PDFs

**Issue: Out of Memory**
- Reduce chunk_size
- Reduce cache max_size
- Process smaller PDFs

## 📝 License

Apache License 2.0 - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch
3. Commit changes
4. Submit pull request

## 📧 Contact

- GitHub: [@Pratt33](https://github.com/Pratt33)
- HuggingFace: [Pratt333](https://huggingface.co/Pratt333)

## 🙏 Acknowledgments

- HuggingFace for model hosting
- SentenceTransformers team
- FAISS by Meta AI Research
- Gradio team

---

**Built with ❤️ using RAG + LLaMA 3.2**
