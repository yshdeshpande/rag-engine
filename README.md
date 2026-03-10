# RAG Engine — Production-Grade Retrieval-Augmented Generation with Evaluation

A retrieval-augmented generation system built from scratch (no LangChain), featuring hybrid retrieval, cross-encoder re-ranking, multiple chunking strategies, and a quantitative evaluation framework with ablation studies.

## Architecture

```mermaid
flowchart TB
    subgraph Ingestion["1 — Data Ingestion"]
        RAW["Raw Corpus<br/>(PDF / HTML / TXT)"]
        LOADER["Document Loader"]
        CLEANER["Text Cleaner<br/>& Metadata Extractor"]
        RAW --> LOADER --> CLEANER
    end

    subgraph Chunking["2 — Chunking (pluggable)"]
        FIXED["Fixed-size"]
        SENT["Sentence-based"]
        SEM["Semantic"]
        REC["Recursive"]
    end

    CLEANER --> Chunking

    subgraph Indexing["3 — Embedding & Indexing"]
        DENSE_EMB["Dense Embeddings<br/>(BGE-large / E5)"]
        FAISS["FAISS Index"]
        BM25["BM25 Index"]
        DENSE_EMB --> FAISS
    end

    Chunking --> DENSE_EMB
    Chunking --> BM25

    subgraph Retrieval["4 — Hybrid Retrieval"]
        DENSE_RET["Dense Retrieval"]
        SPARSE_RET["Sparse Retrieval (BM25)"]
        RRF["Reciprocal Rank Fusion"]
        RERANKER["Cross-Encoder Re-ranker"]
        DENSE_RET --> RRF
        SPARSE_RET --> RRF
        RRF --> RERANKER
    end

    FAISS --> DENSE_RET
    BM25 --> SPARSE_RET

    subgraph Generation["5 — Generation"]
        PROMPT["Prompt Builder<br/>(with citations)"]
        LLM["LLM<br/>(Claude / Mistral)"]
        PROMPT --> LLM
    end

    RERANKER --> PROMPT

    subgraph Evaluation["6 — Evaluation Framework"]
        BENCH["Benchmark Dataset<br/>(100-200 Q&A pairs)"]
        RET_METRICS["Retrieval Metrics<br/>Recall@K · MRR · NDCG"]
        GEN_METRICS["Generation Metrics<br/>Faithfulness · Hallucination"]
        SYS_METRICS["System Metrics<br/>Latency · Throughput"]
    end

    RERANKER --> RET_METRICS
    LLM --> GEN_METRICS
    LLM --> SYS_METRICS

    QUERY(["User Query"]) --> DENSE_RET
    QUERY --> SPARSE_RET
    LLM --> ANSWER(["Answer with Citations"])

    style Ingestion fill:#1a1a2e,stroke:#e94560,color:#eee
    style Chunking fill:#1a1a2e,stroke:#0f3460,color:#eee
    style Indexing fill:#1a1a2e,stroke:#0f3460,color:#eee
    style Retrieval fill:#1a1a2e,stroke:#e94560,color:#eee
    style Generation fill:#1a1a2e,stroke:#0f3460,color:#eee
    style Evaluation fill:#1a1a2e,stroke:#e94560,color:#eee
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/rag-engine.git
cd rag-engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Download corpus
python scripts/download_data.py

# Run ingestion pipeline
python scripts/run_ingestion.py

# Run evaluation
python scripts/run_evaluation.py
```

## Project Structure

```
src/
├── ingestion/      # Document loaders, cleaners, metadata extraction
├── chunking/       # Pluggable chunking strategies (fixed, sentence, semantic, recursive)
├── retrieval/      # Dense, sparse, hybrid retrieval + re-ranker
├── generation/     # Prompt templates, LLM generation with citations
├── evaluation/     # Retrieval & generation metrics, benchmark runner
└── pipeline.py     # End-to-end orchestration
```

## Results

> _Results tables and charts will be added as experiments are completed._

## Lessons Learned

> _To be updated as the project progresses._

## Future Work

- Query decomposition for multi-hop questions
- Contextual compression of retrieved passages
- Fine-tuned embedding models on domain-specific data
- Streaming generation with FastAPI WebSockets
