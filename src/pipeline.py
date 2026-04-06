"""End-to-end RAG pipeline orchestration.

Wires together ingestion, chunking, retrieval, and generation into a
single configurable pipeline driven by configs/default.yaml.

Usage:
    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_yaml("configs/default.yaml")
    pipeline.ingest()          # load, clean, chunk, index
    result = pipeline.query("What did the court hold in ...?")
    print(result.answer)
"""

from pathlib import Path

import yaml

from src.chunking.base import BaseChunker, Chunk
from src.chunking.fixed import FixedChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.semantic import SemanticChunker
from src.chunking.sentence import SentenceChunker
from src.generation.generator import GenerationResult, RAGGenerator
from src.ingestion.cleaner import clean_batch
from src.ingestion.loader import Document, load_directory
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever


class RAGPipeline:
    """Configurable RAG pipeline: ingest -> chunk -> index -> query."""

    def __init__(
        self,
        chunker: BaseChunker,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        hybrid_retriever: HybridRetriever,
        generator: RAGGenerator,
        config: dict,
    ):
        self.chunker = chunker
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.hybrid = hybrid_retriever
        self.generator = generator
        self.config = config

        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []

    @classmethod
    def from_yaml(cls, config_path: str) -> "RAGPipeline":
        """Build a pipeline from a YAML config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        chunker = _build_chunker(config["chunking"])
        dense = DenseRetriever(
            model_name=config["embedding"]["model"],
            batch_size=config["embedding"]["batch_size"],
        )
        sparse = SparseRetriever()

        retrieval_cfg = config["retrieval"]
        hybrid = HybridRetriever(
            dense=dense,
            sparse=sparse,
            dense_weight=retrieval_cfg["dense_weight"],
            sparse_weight=retrieval_cfg["sparse_weight"],
        )

        gen_cfg = config["generation"]
        generator = RAGGenerator(
            provider=gen_cfg["provider"],
            model=gen_cfg["model"],
            max_tokens=gen_cfg["max_tokens"],
            temperature=gen_cfg["temperature"],
            base_url=gen_cfg.get("base_url", "http://localhost:11434"),
        )

        return cls(
            chunker=chunker,
            dense_retriever=dense,
            sparse_retriever=sparse,
            hybrid_retriever=hybrid,
            generator=generator,
            config=config,
        )

    def ingest(self, directory: str | Path | None = None) -> None:
        """Load documents, clean, chunk, and build retrieval indices.

        If directory is not provided, uses corpus.processed_dir from config.
        """
        if directory is None:
            directory = Path(self.config["corpus"]["processed_dir"])
        else:
            directory = Path(directory)

        # Load and clean
        print(f"Loading documents from {directory}...")
        self.documents = load_directory(directory)
        print(f"Loaded {len(self.documents)} documents")

        self.documents = clean_batch(self.documents)

        # Chunk all documents
        self.chunks = []
        for doc in self.documents:
            doc_chunks = self.chunker.chunk(
                text=doc.text,
                doc_id=doc.doc_id,
                metadata=doc.metadata,
            )
            self.chunks.extend(doc_chunks)
        print(f"Created {len(self.chunks)} chunks")

        # Build retrieval indices
        print("Building dense index...")
        self.dense.index_chunks(self.chunks)

        print("Building sparse index...")
        self.sparse.index(self.chunks)
        print("Indexing complete")

    def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> GenerationResult:
        """Run the full RAG pipeline: retrieve -> generate.

        Returns a GenerationResult with the answer, model info, and context.
        """
        if not self.chunks:
            raise RuntimeError("No chunks indexed. Call pipeline.ingest() first.")

        top_k = top_k or self.config["retrieval"]["top_k"]
        retrieved = self.hybrid.search(question, top_k=top_k)

        return self.generator.generate(question, retrieved)

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Run retrieval only (useful for evaluation and debugging)."""
        if not self.chunks:
            raise RuntimeError("No chunks indexed. Call pipeline.ingest() first.")

        top_k = top_k or self.config["retrieval"]["top_k"]
        return self.hybrid.search(question, top_k=top_k)


def _build_chunker(chunking_cfg: dict) -> BaseChunker:
    """Instantiate the right chunker based on config."""
    strategy = chunking_cfg["strategy"]

    if strategy == "fixed":
        cfg = chunking_cfg["fixed"]
        return FixedChunker(
            chunk_size=cfg["chunk_size"],
            overlap=cfg["overlap"],
        )
    elif strategy == "sentence":
        cfg = chunking_cfg["sentence"]
        return SentenceChunker(max_chunk_size=cfg["max_sentences"])
    elif strategy == "recursive":
        cfg = chunking_cfg["recursive"]
        return RecursiveChunker(
            max_chunk_size=cfg["chunk_size"],
            separators=cfg.get("separators"),
        )
    elif strategy == "semantic":
        cfg = chunking_cfg["semantic"]
        return SemanticChunker(
            model_name=cfg["embedding_model"],
            similarity_threshold=cfg["similarity_threshold"],
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
