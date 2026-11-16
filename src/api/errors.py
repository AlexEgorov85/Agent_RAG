"""Единый формат ошибок публичного API.

Сообщения фиксированы и не содержат traceback, prompt, ключей или тел ответов
провайдера. Исключение ``ApiError`` перехватывается обработчиком приложения.
"""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """HTTP-ошибка контракта §7.8."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        request_id: str | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        self.retryable = retryable
        self.details = details or {}

    def payload(self) -> dict[str, Any]:
        """Тело ошибки в едином конверте."""
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": bool(self.retryable),
        }
        if self.request_id is not None:
            error["request_id"] = self.request_id
        if self.details:
            error["details"] = self.details
        return {"error": error}


class BackendUnavailable(ApiError):
    """Backend RAG недоступен (503)."""
    def __init__(self, message: str = "Backend RAG недоступен.") -> None:
        super().__init__(503, "backend_unavailable", message, retryable=True)


class RequestInProgress(ApiError):
    """У исследования уже есть активный запрос (409)."""
    def __init__(self, active_request_id: str) -> None:
        super().__init__(
            409,
            "request_in_progress",
            "В этом диалоге уже выполняется запрос.",
            retryable=True,
            details={"active_request_id": active_request_id},
        )


def validation_error(message: str = "Некорректные параметры запроса.") -> ApiError:
    """Ошибка валидации 422."""
    return ApiError(422, "validation_error", message)


def not_found(message: str = "Объект не найден.") -> ApiError:
    """Ошибка 404 для отсутствующего объекта."""
    return ApiError(404, "not_found", message)


def request_not_found(request_id: str | None = None) -> ApiError:
    """Ошибка 404 для отсутствующего запроса."""
    return ApiError(
        404,
        "request_not_found",
        "Запрос не найден.",
        request_id=request_id,
    )
