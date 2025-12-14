"""
Lightweight public interface for the RAG agent package.

Design goals:
- Avoid heavy imports at package import time (no vector DB or LLM side‑effects).
- Keep `__all__` small and explicit.
- Let callers import heavy components directly from their submodules.
"""

from .rag_agent import RAGAgent

__all__ = [
    "RAGAgent",
    "rag_agent",  # provided lazily via __getattr__
]


def __getattr__(name: str):
    """Lazy access to the global `rag_agent` instance without eager imports."""
    if name == "rag_agent":
        from .rag_agent import rag_agent  # local import to avoid circulars

        return rag_agent
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

