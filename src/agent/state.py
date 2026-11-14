"""Типизированные публичные контракты и состояние графа агента отчётов."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from src.retrieval.hybrid_retriever import SearchFilters


ToolName = Literal[
    "search_reports", "get_structure_tree", "get_entities", "get_chunk"
]
ReasonCode = Literal[
    "fact_lookup", "clarify_structure", "explore_entities", "inspect_evidence"
]
AnswerStatus = Literal["answer", "abstain"]


@dataclass(frozen=True)
class QueryPlan:
    """Проверенный результат роутера; явные фильтры API остаются главными."""

    intent: Literal["factoid", "structural", "temporal", "multi_hop"] = "factoid"
    years: list[int] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    section_filter: str | None = None
    entities: list[str] = field(default_factory=list)
    needs_rewriting: bool = False
    needs_graph: bool = False
    answer_mode: Literal["concise", "comparison", "outline"] = "concise"


@dataclass(frozen=True)
class AgentAction:
    """Один вызов инструмента из белого списка, предложенный селектором действий."""

    tool_name: str
    arguments: dict[str, Any]
    reason_code: str


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    doc_id: str | None = None
    year: int | None = None
    issuer: str | None = None
    page_pdf: int | None = None
    page_label: str | None = None
    section_path: str | None = None
    quote: str | None = None


@dataclass(frozen=True)
class Answer:
    """Контракт ответа из spec_target.md, проверенный по манифесту запуска."""

    status: AnswerStatus
    answer: str
    citations: list[Citation] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ToolEvent:
    tool_name: str
    arguments: dict[str, Any]
    status: str
    result_count: int = 0
    duration_ms: float = 0.0
    warning: str | None = None


@dataclass(frozen=True)
class AgentEvent:
    """Безопасное для UI потоковое событие; ``result`` заполняется на run_completed."""

    event_type: str
    request_id: str
    data: dict[str, Any] = field(default_factory=dict)
    result: AgentRunResult | None = None


@dataclass(frozen=True)
class AgentRunResult:
    request_id: str
    conversation_id: str
    answer: Answer
    tool_events: list[ToolEvent]
    effective_filters: SearchFilters
    warnings: list[str]
    timings: dict[str, float]
    status: Literal["completed", "abstained", "failed"] = "completed"


class AgentState(TypedDict, total=False):
    """Каналы StateGraph для одного изолированного пользовательского хода."""

    request_id: str
    conversation_id: str
    question: str
    history: list[dict[str, str]]
    filters: SearchFilters
    effective_filters: SearchFilters
    query_plan: QueryPlan | None
    action: AgentAction | None
    tool_events: list[ToolEvent]
    retrieval_results: list[Any]
    evidence_manifest: set[str]
    search_evidence_manifest: set[str]
    graph_evidence_manifest: set[str]
    evidence: dict[str, Any]
    answer: Answer | None
    status: str
    tool_calls_used: int
    search_calls_used: int
    answer_repairs_used: int
    warnings: list[str]
    timings: dict[str, float]
    unknown_action_fallback_used: bool
    messages: list[Any]
