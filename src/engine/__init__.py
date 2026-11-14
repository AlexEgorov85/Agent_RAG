"""RAG answering layer: whole-chunk context, prompts, generation, and engine."""

from src.engine.context import (
    CONTEXT_FIELDS,
    ContextManifest,
    PackedContext,
    pack_context,
)
from src.engine.generator import AnswerGenerator
from src.engine.prompts import SYSTEM_PROMPT, build_user_prompt
from src.engine.rag_engine import RAGEngine, build_engine

__all__ = [
    "AnswerGenerator",
    "CONTEXT_FIELDS",
    "ContextManifest",
    "PackedContext",
    "RAGEngine",
    "SYSTEM_PROMPT",
    "build_engine",
    "build_user_prompt",
    "pack_context",
]
