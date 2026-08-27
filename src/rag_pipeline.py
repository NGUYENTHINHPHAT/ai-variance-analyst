"""
RAG Pipeline — Retrieval-Augmented Generation for Financial Context
Embeds company documents and retrieves relevant context for variance explanations.
"""

import os
import json
from pathlib import Path
import numpy as np

# ── Graceful imports ──────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


RAG_DOCS_DIR   = "rag_docs"
CHROMA_DIR     = "data/chroma_db"
COLLECTION_NAME= "novatech_financial_context"
EMBED_MODEL    = "all-MiniLM-L6-v2"
TOP_K          = 5
CHUNK_SIZE     = 600   # characters
CHUNK_OVERLAP  = 100


def chunk_document(text: str, source: str) -> list[dict]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({
                "text": chunk.strip(),
                "source": source,
                "chunk_id": f"{source}_{len(chunks)}",
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_rag_documents(docs_dir: str = RAG_DOCS_DIR) -> list[dict]:
    """Load all markdown documents from rag_docs directory."""
    docs = []
    for path in Path(docs_dir).glob("*.md"):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_document(text, path.stem)
        docs.extend(chunks)
    print(f"  Loaded {len(docs)} chunks from {len(list(Path(docs_dir).glob('*.md')))} documents")
    return docs


def build_vector_store(docs: list[dict], persist_dir: str = CHROMA_DIR):
    """Build and persist ChromaDB vector store."""
    if not HAS_ST or not HAS_CHROMA:
        print("  ⚠ ChromaDB or SentenceTransformers not available. Using fallback keyword search.")
        return None

    os.makedirs(persist_dir, exist_ok=True)
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=persist_dir)

    # Reset collection
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Explicit cosine distance -- sentence-transformer embeddings are best compared
    # by direction, not magnitude. Chroma's default is squared L2, which happens to
    # rank similarly for MiniLM's roughly-uniform-magnitude output but isn't the
    # metric actually intended here.
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    texts    = [d["text"]     for d in docs]
    ids      = [d["chunk_id"] for d in docs]
    metadatas= [{"source": d["source"]} for d in docs]

    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"  ✅ Vector store built: {len(docs)} chunks indexed")
    return collection


def keyword_search_fallback(query: str, docs: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Simple keyword-based retrieval when ChromaDB is not available."""
    query_terms = set(query.lower().split())
    scored = []
    for doc in docs:
        doc_terms = set(doc["text"].lower().split())
        score = len(query_terms & doc_terms)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


class FinancialRAG:
    """Main RAG interface for financial context retrieval."""

    def __init__(self, docs_dir: str = RAG_DOCS_DIR, persist_dir: str = CHROMA_DIR):
        self.docs = load_rag_documents(docs_dir)
        self.collection = None
        self.model = None

        if HAS_ST and HAS_CHROMA:
            self.model = SentenceTransformer(EMBED_MODEL)
            client = chromadb.PersistentClient(path=persist_dir)
            try:
                self.collection = client.get_collection(COLLECTION_NAME)
                print(f"  ✅ Loaded existing vector store ({self.collection.count()} chunks)")
            except Exception:
                print("  Building vector store...")
                self.collection = build_vector_store(self.docs, persist_dir)
        else:
            print("  Using keyword search fallback (install sentence-transformers + chromadb for embeddings)")

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        """Retrieve relevant context for a given variance query."""
        if self.collection and self.model:
            query_embedding = self.model.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
            )
            chunks = results["documents"][0]
            sources = [m["source"] for m in results["metadatas"][0]]
        else:
            results = keyword_search_fallback(query, self.docs, top_k)
            chunks = [r["text"] for r in results]
            sources = [r["source"] for r in results]

        context_parts = []
        for chunk, source in zip(chunks, sources):
            context_parts.append(f"[Source: {source}]\n{chunk}")

        return "\n\n---\n\n".join(context_parts)

    def retrieve_for_variance(self, account: str, department: str,
                               period: str, variance_pct: float) -> str:
        """Build a targeted retrieval query from variance details."""
        direction = "overspend" if variance_pct > 0 else "underspend"
        query = (
            f"{account} {department} {direction} variance "
            f"budget actual {period} driver explanation seasonality"
        )
        return self.retrieve(query)


if __name__ == "__main__":
    print("Building RAG index...")
    rag = FinancialRAG()
    
    test_query = "Marketing advertising overspend Q3 July August"
    print(f"\nTest retrieval for: '{test_query}'")
    context = rag.retrieve(test_query, top_k=2)
    print(context[:500] + "...")
