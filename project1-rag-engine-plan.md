# Project 1: Production-Grade RAG Engine with Evaluation Framework

## Context

I'm an AI Engineer with 1.5 years of experience at UST Global, preparing to switch to a product company (Google, Microsoft, Salesforce, Atlassian) in an AI/ML role. This is the first of 4 portfolio projects I'm building over 6 months to strengthen my GitHub profile. I have a B.E. in Mechanical Engineering from BITS Goa. I'm based in India.

The goal of this project is to demonstrate that I understand how real AI products work — not just a basic RAG chatbot, but a production-grade system with proper evaluation, experimentation, and engineering quality.

---

## Project Overview

Build a retrieval-augmented generation system that goes well beyond a basic LangChain tutorial. The system should:

1. Ingest a large, messy real-world corpus
2. Implement and compare multiple chunking strategies empirically
3. Use hybrid retrieval combining dense embeddings + sparse retrieval (BM25)
4. Include a re-ranker stage
5. Have a proper evaluation harness with quantitative metrics
6. Be documented with experiment results, tables, and charts

---

## Suggested Corpus Options (pick one)

- **Indian Supreme Court judgments** — publicly available, messy formatting, real-world relevance
- **ArXiv papers in a specific ML sub-domain** — technical, well-structured, large volume
- **Indian Parliament debates / Lok Sabha Q&A** — interesting, publicly available
- **Wikipedia subset on a specific domain** (e.g., Indian history, medical topics)

Choose something with enough volume (10k+ documents) and messiness to make the retrieval challenge non-trivial.

---

## Architecture & Components

### 1. Data Ingestion Pipeline
- Document loader supporting multiple formats (PDF, HTML, plain text)
- Text cleaning and preprocessing (handle OCR artifacts, encoding issues, metadata extraction)
- Store raw + processed documents with metadata

### 2. Chunking Module (Experiment with multiple strategies)
- Fixed-size chunking (with overlap)
- Sentence-based chunking
- Semantic chunking (using embedding similarity to find natural breakpoints)
- Recursive/hierarchical chunking
- Document-structure-aware chunking (headings, paragraphs, sections)
- **Each strategy should be a pluggable component so you can swap and compare**

### 3. Embedding & Indexing
- Dense embeddings: Use models like BGE-large, E5, or ColBERT
- Sparse retrieval: BM25 (using rank-bm25 or Elasticsearch)
- Vector store: FAISS or ChromaDB for dense vectors
- Hybrid retrieval: Combine dense + sparse scores (experiment with fusion methods like Reciprocal Rank Fusion)

### 4. Re-Ranker
- Cross-encoder re-ranker (e.g., ms-marco-MiniLM or BGE-reranker)
- Compare retrieval quality with and without re-ranking

### 5. Generation
- Use an open-source LLM (Mistral 7B, Llama 3.1 8B) or API-based model
- Implement proper prompt templates with retrieved context
- Add citation/source attribution in generated answers
- Handle edge cases: no relevant context found, contradictory sources, very long contexts

### 6. Evaluation Framework (THIS IS THE DIFFERENTIATOR)
- **Create a benchmark dataset**: 100-200 question-answer pairs with ground truth
  - Some manually curated
  - Some generated via LLM and validated
- **Retrieval metrics**: Recall@K, Precision@K, MRR, NDCG
- **Generation metrics**:
  - Answer faithfulness (does the answer stick to retrieved context?) — use LLM-as-judge
  - Answer relevance (does it actually answer the question?)
  - Hallucination rate detection
  - Factual accuracy against ground truth
- **System metrics**: Latency (retrieval, re-ranking, generation), throughput, cost per query
- **Ablation studies**: Measure impact of each component (chunking strategy, hybrid vs dense-only, with/without re-ranker)
- Present results in tables and charts in the README

---

## Tech Stack (Suggested)

- **Language**: Python 3.10+
- **Embeddings**: sentence-transformers, FlagEmbedding (for BGE)
- **Vector Store**: FAISS or ChromaDB
- **Sparse Retrieval**: rank-bm25 or Elasticsearch
- **Re-ranker**: cross-encoder models via sentence-transformers
- **LLM**: HuggingFace transformers, or API calls (Anthropic/OpenAI for evaluation)
- **Orchestration**: Custom pipeline (avoid heavy LangChain abstraction — write your own to show understanding)
- **Experiment Tracking**: Simple JSON/CSV logs, or Weights & Biases
- **Visualization**: Matplotlib/Plotly for charts in README
- **Testing**: pytest
- **Containerization**: Dockerfile for reproducibility

---

## Repo Structure (Suggested)

```
rag-engine/
├── README.md                  # Detailed with architecture diagram, results, lessons learned
├── pyproject.toml             # or requirements.txt
├── Dockerfile
├── Makefile                   # Common commands: ingest, evaluate, serve
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py          # Document loaders for different formats
│   │   ├── cleaner.py         # Text cleaning and preprocessing
│   │   └── metadata.py        # Metadata extraction
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract base class for chunkers
│   │   ├── fixed.py           # Fixed-size chunking
│   │   ├── sentence.py        # Sentence-based chunking
│   │   ├── semantic.py        # Semantic chunking
│   │   └── recursive.py       # Recursive chunking
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense.py           # Dense embedding retrieval
│   │   ├── sparse.py          # BM25 retrieval
│   │   ├── hybrid.py          # Hybrid fusion
│   │   └── reranker.py        # Cross-encoder re-ranking
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompts.py         # Prompt templates
│   │   └── generator.py       # LLM generation with context
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── retrieval_metrics.py   # Recall, Precision, MRR, NDCG
│   │   ├── generation_metrics.py  # Faithfulness, relevance, hallucination
│   │   ├── system_metrics.py      # Latency, throughput
│   │   └── benchmark.py           # Run full evaluation suite
│   └── pipeline.py            # End-to-end RAG pipeline
├── configs/
│   └── default.yaml           # Configuration for experiments
├── data/
│   ├── raw/                   # Raw downloaded corpus
│   ├── processed/             # Cleaned documents
│   └── benchmark/             # Evaluation Q&A pairs
├── experiments/
│   └── results/               # Experiment logs, metrics, charts
├── notebooks/
│   ├── data_exploration.ipynb
│   └── results_analysis.ipynb
├── scripts/
│   ├── download_data.py
│   ├── run_ingestion.py
│   ├── run_evaluation.py
│   └── generate_charts.py
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_evaluation.py
└── serve/
    └── api.py                 # FastAPI endpoint to query the RAG system
```

---

## Week-by-Week Plan (~5-6 weeks)

### Week 1: Setup + Data Ingestion
- Initialize repo with proper structure, README skeleton, .gitignore, pyproject.toml
- Choose and download corpus
- Build data ingestion pipeline (loaders, cleaners)
- Commit daily with meaningful messages
- **Deliverable**: Raw corpus ingested and cleaned, documented in README

### Week 2: Chunking + Embedding
- Implement 3-4 chunking strategies as pluggable modules
- Set up dense embedding pipeline (choose model, build index)
- Set up BM25 sparse retrieval
- Write unit tests for chunking
- **Deliverable**: Documents chunked, embedded, and indexed

### Week 3: Retrieval + Re-ranking
- Implement hybrid retrieval with score fusion
- Add cross-encoder re-ranker
- Build the end-to-end retrieval pipeline
- Basic retrieval testing with sample queries
- **Deliverable**: Working retrieval pipeline, can query and get ranked results

### Week 4: Generation + Benchmark Creation
- Integrate LLM generation with retrieved context
- Build prompt templates with citation support
- Create benchmark dataset (100-200 Q&A pairs)
- **Deliverable**: Full RAG pipeline working end-to-end, benchmark dataset ready

### Week 5: Evaluation Framework + Experiments
- Implement all retrieval metrics (Recall@K, MRR, NDCG)
- Implement generation metrics (faithfulness, hallucination, relevance)
- Run ablation studies across configurations
- Generate charts and results tables
- **Deliverable**: Complete evaluation results, charts ready

### Week 6: Polish + Documentation + API
- Build FastAPI serving endpoint
- Add Dockerfile
- Write comprehensive README with architecture diagram, results, lessons learned
- Clean up code, add docstrings and type hints everywhere
- Final review and polish
- **Deliverable**: Production-ready repo that impresses reviewers

---

## README Must-Haves (What Impresses Reviewers)

1. **Problem Statement**: Why this matters, not just what it does
2. **Architecture Diagram**: Visual overview of the pipeline (use Mermaid or a PNG)
3. **Quick Start**: How to run it in < 5 commands
4. **Results Section**: Tables comparing chunking strategies, retrieval methods, with/without re-ranker
5. **Charts**: Recall@K curves, latency distributions, ablation results
6. **Lessons Learned**: What failed, what surprised you, what you'd do differently
7. **Future Work**: Shows you think beyond the current scope

---

## Key Principles

- **Commit daily** with meaningful commit messages (not "update", but "add semantic chunking with cosine similarity breakpoints")
- **Write clean, modular code** with type hints, docstrings, and tests
- **Avoid over-relying on LangChain** — write your own pipeline to demonstrate understanding
- **Quantify everything** — numbers and charts are what separate this from tutorial projects
- **Document as you go** — don't leave README for the end

---

## How This Maps to Interview Talking Points

- "I implemented hybrid retrieval and measured a X% improvement in Recall@10 over dense-only"
- "Semantic chunking outperformed fixed-size by Y% on faithfulness but was Z ms slower"
- "The re-ranker improved MRR from A to B but added C ms latency — here's the tradeoff analysis"
- "I built an LLM-as-judge evaluation pipeline to measure hallucination rates across configurations"

These are exactly the kinds of concrete, data-backed statements that impress interviewers at Google, Microsoft, Salesforce, and Atlassian.
