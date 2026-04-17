"""FastAPI endpoint to query the RAG system.

Exposes the RAG pipeline as a REST API. Start with:
    uvicorn serve.api:app --reload

Endpoints:
    POST /query    — ask a question, get an answer with sources
    POST /retrieve — retrieve relevant chunks without generating
    GET  /health   — liveness check
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline import RAGPipeline

app = FastAPI(
    title="RAG Engine",
    description="Production-grade Retrieval-Augmented Generation API",
    version="0.1.0",
)

# Global pipeline instance — initialized via /ingest or on startup
_pipeline: RAGPipeline | None = None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")


class Source(BaseModel):
    doc_id: str
    chunk_index: int
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    model: str
    sources: list[Source]
    usage: dict


class RetrieveResponse(BaseModel):
    sources: list[Source]


class IngestRequest(BaseModel):
    config_path: str = Field(default="configs/default.yaml")
    data_dir: str | None = Field(default=None, description="Override data directory")


class IngestResponse(BaseModel):
    num_documents: int
    num_chunks: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_loaded": _pipeline is not None and len(_pipeline.chunks) > 0,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """Load documents, chunk, and build retrieval indices."""
    global _pipeline

    config_path = Path(request.config_path)
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {config_path}")

    _pipeline = RAGPipeline.from_yaml(str(config_path))
    _pipeline.ingest(directory=request.data_dir)

    return IngestResponse(
        num_documents=len(_pipeline.documents),
        num_chunks=len(_pipeline.chunks),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Run the full RAG pipeline: retrieve + generate."""
    if _pipeline is None or not _pipeline.chunks:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Call /ingest first.")

    result = _pipeline.query(request.question, top_k=request.top_k)

    # Extract source info from the retrieved chunks used in generation
    retrieved = _pipeline.retrieve(request.question, top_k=request.top_k)
    sources = [
        Source(
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text[:500],
            score=round(score, 4),
        )
        for chunk, score in retrieved
    ]

    return QueryResponse(
        answer=result.answer,
        model=result.model,
        sources=sources,
        usage=result.usage,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: QueryRequest):
    """Retrieve relevant chunks without generating an answer."""
    if _pipeline is None or not _pipeline.chunks:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Call /ingest first.")

    results = _pipeline.retrieve(request.question, top_k=request.top_k)
    sources = [
        Source(
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text[:500],
            score=round(score, 4),
        )
        for chunk, score in results
    ]

    return RetrieveResponse(sources=sources)
