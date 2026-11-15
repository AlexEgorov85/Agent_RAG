"""Pydantic-модели публичного контракта (snake_case, ISO UTC, UUID).

Модели описывают только транспорт. Бизнес-логика, retrieval и генерация
остаются в Python-ядре; adapter не «исправляет» значения и не выдумывает поля.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

SourceType = Literal["text", "table"]
MessageStatus = Literal[
    "pending", "streaming", "completed", "abstained", "failed", "cancelled"
]
ResearchStatus = Literal[
    "idle", "running", "completed", "abstained", "failed", "cancelled"
]
RequestStatus = Literal[
    "queued", "running", "completed", "abstained", "failed", "cancelled"
]
AnswerStatus = Literal["answer", "abstain"]

_Year = Annotated[StrictInt, Field(ge=1900, le=2100)]


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _clean_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of strings")
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("expected a list of strings")
        text = item.strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned or None


class FilterSet(ApiModel):
    """Публичные фильтры запроса; пустые значения не передаются."""

    years: list[_Year] | None = None
    year_from: _Year | None = None
    year_to: _Year | None = None
    issuers: list[str] | None = None
    allowed_doc_ids: list[str] | None = None
    kinds: list[SourceType] | None = None

    @field_validator("issuers", "allowed_doc_ids", mode="before")
    @classmethod
    def _clean_strings(cls, value: Any) -> list[str] | None:
        return _clean_str_list(value)

    @field_validator("years", "kinds", mode="before")
    @classmethod
    def _dedupe(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            seen: list[Any] = []
            for item in value:
                if item not in seen:
                    seen.append(item)
            return seen or None
        return value

    @model_validator(mode="after")
    def _validate_range(self) -> "FilterSet":
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must be less than or equal to year_to")
        return self


class EvidenceItem(ApiModel):
    evidence_id: str
    chunk_id: str
    document_id: str | None = None
    document_title: str | None = None
    issuer: str | None = None
    year: int | None = None
    section_title: str | None = None
    section_path: list[str] | None = None
    page_start: int | None = None
    page_end: int | None = None
    page_label: str | None = None
    excerpt: str = ""
    source_type: SourceType | None = None
    has_full_text: bool = False


class AnalysisResult(ApiModel):
    calculations: list[dict[str, Any]] = Field(default_factory=list)
    facts: list[dict[str, Any]] = Field(default_factory=list)
    coverage: dict[str, Any] | None = None
    structural_changes: list[dict[str, Any]] | None = None
    verification: dict[str, Any] | None = None


class ErrorInfo(ApiModel):
    code: str
    message: str
    retryable: bool = False


class ChatMessage(ApiModel):
    id: str
    research_id: str
    role: Literal["user", "assistant"]
    content: str = ""
    created_at: str
    status: MessageStatus
    request_id: str | None = None
    effective_filters: FilterSet | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    analysis: AnalysisResult | None = None
    warnings: list[str] = Field(default_factory=list)
    error: ErrorInfo | None = None


class ResearchSummary(ApiModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    last_message_preview: str | None = None
    last_status: ResearchStatus = "idle"
    messages_count: int = 0


class ResearchListResponse(ApiModel):
    items: list[ResearchSummary] = Field(default_factory=list)
    total: int = 0
    next_offset: int | None = None


class ResearchCreate(ApiModel):
    title: str | None = Field(default=None, max_length=200)

    @field_validator("title", mode="before")
    @classmethod
    def _trim(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("title must be a string")
        return value.strip() or None


class ResearchUpdate(ApiModel):
    title: str = Field(min_length=1, max_length=200)

    @field_validator("title", mode="before")
    @classmethod
    def _trim(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("title must be a string")
        return value.strip()


class ResearchRef(ApiModel):
    id: str
    updated_at: str


class MessageCreate(ApiModel):
    content: str = Field(min_length=1, max_length=16_000)
    client_message_id: UUID
    filters: FilterSet | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _trim(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("content must be a string")
        return value.strip()


class MessageListResponse(ApiModel):
    items: list[ChatMessage] = Field(default_factory=list)
    next_before: str | None = None


class MessageAccepted(ApiModel):
    request_id: str
    user_message: ChatMessage
    assistant_message_id: str
    events_url: str
    research: ResearchRef


class RequestState(ApiModel):
    request_id: str
    status: RequestStatus
    message_id: str
    message: ChatMessage | None = None
    error: ErrorInfo | None = None


class CancelResponse(ApiModel):
    request_id: str
    status: Literal["cancelled", "not_supported", "already_finished"]
    cancel_supported: bool
    code: str | None = None


class HealthResponse(ApiModel):
    status: Literal["ok", "unavailable"]
    contract_version: str = "1"
    database: Literal["ok", "error"] = "ok"
    corpus_available: bool = False
    agent_available: bool = False


class CapabilitiesResponse(ApiModel):
    contract_version: str = "1"
    token_streaming: bool
    incremental_evidence: bool
    coverage: bool
    calculations: bool
    structure_comparison: bool
    structure_content_comparison: bool
    graph_panel: bool
    pdf_page_links: bool
    evaluation_ui: bool
    evaluation_runs: bool = False
    evaluation_sse: bool
    evaluation_cancel: bool
    retrieval_modes: list[str]
    judge: bool
    max_repeat: int = 5
    max_queued: int = 3


class CorpusDocument(ApiModel):
    doc_id: str
    year: int | None = None
    issuer: str | None = None
    title: str | None = None


class CorpusResponse(ApiModel):
    name: str
    years: list[int] = Field(default_factory=list)
    documents: list[CorpusDocument] = Field(default_factory=list)
    issuers: list[str] = Field(default_factory=list)
    years_available: bool = False


class EvidenceDetail(ApiModel):
    chunk_id: str
    document_id: str | None = None
    document_title: str | None = None
    issuer: str | None = None
    year: int | None = None
    section_path: list[str] | None = None
    page_start: int | None = None
    page_end: int | None = None
    page_label: str | None = None
    source_type: SourceType | None = None
    text: str = ""
    html: str | None = None


class ErrorDetail(ApiModel):
    code: str
    message: str
    request_id: str | None = None
    retryable: bool = False
    details: dict[str, Any] | None = None


class ErrorResponse(ApiModel):
    error: ErrorDetail
