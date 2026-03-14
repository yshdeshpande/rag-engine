"""Sentence-based chunking using NLTK sentence tokenizer."""

import nltk

from src.chunking.base import BaseChunker, Chunk

nltk.download("punkt_tab", quiet=True)


class SentenceChunker(BaseChunker):

    def __init__(
        self,
        max_chunk_size: int = 512,
        overlap_sentences: int = 2,
    ):
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> list[Chunk]:
        if not text:
            return []

        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_sentences = []
        current_size = 0
        chunk_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_len = len(sentence)

            # If adding this sentence exceeds the limit and we have something, flush
            if current_size + sentence_len > self.max_chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                start_char = text.find(current_sentences[0])
                end_char = start_char + len(chunk_text)

                chunks.append(Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=metadata or {},
                ))
                chunk_index += 1

                # Overlap: carry over the last N sentences
                overlap = current_sentences[-self.overlap_sentences:]
                current_sentences = overlap + [sentence]
                current_size = sum(len(s) for s in current_sentences)
            else:
                current_sentences.append(sentence)
                current_size += sentence_len

        # Flush remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            start_char = text.find(current_sentences[0])
            end_char = start_char + len(chunk_text)

            chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                metadata=metadata or {},
            ))

        return chunks
