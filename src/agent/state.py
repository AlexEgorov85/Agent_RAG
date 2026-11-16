"""Типизированные публичные контракты и состояние графа агента отчётов."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from src.retrieval.hybrid_retriever import SearchFilters


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
    #: До 4 поисковых аспектов и до 5 канонических metric id.
    aspects: list[str] = field(default_factory=list)
    required_metrics: list[str] = field(default_factory=list)
    requires_comparison: bool = False


@dataclass(frozen=True)
class AgentAction:
    """Один вызов инструмента из белого списка, предложенный селектором действий."""

    tool_name: str
    arguments: dict[str, Any]
    reason_code: str


@dataclass(frozen=True)
class Citation:
    """Ссылка на чанк-источник в ответе."""
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
    #: Детерминированные расчёты по ``facts``; не являются фактами из отчёта.
    calculations: list[dict[str, Any]] = field(default_factory=list)
    #: Результат проверки citations/facts/периодов; заполняется AgentService.
    verification: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolEvent:
    """Безопасная запись одного вызова инструмента."""
    tool_name: str
    arguments: dict[str, Any]
    status: str
    result_count: int = 0
    duration_ms: float = 0.0
    warning: str | None = None
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentEvent:
    """Безопасное для UI потоковое событие; ``result`` заполняется на терминальном событии ``run_completed``/``run_failed``."""

    event_type: str
    request_id: str
    data: dict[str, Any] = field(default_factory=dict)
    result: AgentRunResult | None = None


@dataclass(frozen=True)
class AgentRunResult:
    """Полный результат одного хода агента."""
    request_id: str
    conversation_id: str
    answer: Answer
    tool_events: list[ToolEvent]
    effective_filters: SearchFilters
    warnings: list[str]
    timings: dict[str, float]
    status: Literal["completed", "abstained", "failed"] = "completed"
    structural_changes: list[dict[str, Any]] = field(default_factory=list)
    calculations: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def retrieval_diagnostics(self) -> list[dict[str, Any]]:
        """Устаревший alias на ``diagnostics['retrieval_calls']``."""
        value = self.diagnostics.get("retrieval_calls", [])
        return list(value) if isinstance(value, list) else []


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
    #: Текст уточняющего вопроса, если период запрошен неявно (clarify-политика).
    #: При непустом значении поиск не запускается, а ответ — abstain с причиной.
    clarification: str | None
    status: str
    tool_calls_used: int
    search_calls_used: int
    answer_repairs_used: int
    warnings: list[str]
    timings: dict[str, float]
    unknown_action_fallback_used: bool
    messages: list[Any]
    #: Полнота собранных доказательств по периодам.
    expected_years: list[int]
    covered_years: list[int]
    missing_years: list[int]
    coverage: Any
    rewritten_queries: list[str]
    search_attempts: list[dict[str, Any]]
    available_documents: dict[int, list[str]]
    #: Temporal Structure Analysis: деревья разделов и найденные изменения.
    structure_documents: dict[int, str]
    structure_snapshots: dict[int, list[dict[str, Any]]]
    section_matches: list[dict[str, Any]]
    structural_changes: list[dict[str, Any]]
    content_changes: list[dict[str, Any]]
    structure_comparison_complete: bool
    structure_content_checked: bool
