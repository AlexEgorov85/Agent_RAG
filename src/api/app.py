"""ASGI-приложение adapter'а: FastAPI + SSE поверх существующего AgentService.

Импорт модуля не читает env, не создаёт SQLite и не строит индексы. Всё
инициализируется в lifespan; ``AgentService`` создаётся лениво при первом
обращении к backend'у.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.bootstrap import ensure_app_state
from src.api.config import DEFAULT_CORS_ORIGINS, ApiConfig
from src.api.errors import ApiError
from src.api.routes import router as api_router
from src.api.routes.documents import router as documents_router
from src.api.routes.evaluation import (
    ensure_evaluation_state,
    router as evaluation_router,
)
from src.api.services import Services
from src.api.store import Store

_MESSAGES = {
    400: "Некорректный запрос.",
    403: "Доступ запрещён.",
    404: "Объект не найден.",
    405: "Метод не поддерживается.",
    410: "Ресурс больше недоступен.",
    413: "Превышен допустимый размер.",
    422: "Некорректные параметры запроса.",
    429: "Превышен лимит запросов.",
    500: "Внутренняя ошибка.",
    503: "Backend недоступен.",
}


def _error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: dict | None = None,
) -> JSONResponse:
    error: dict = {"code": code, "message": message, "retryable": retryable}
    if details:
        error["details"] = details
    return JSONResponse(status_code=status_code, content={"error": error})


def _mount_spa(app: FastAPI, config: ApiConfig | None) -> None:
    """Опциональная раздача `frontend/dist` тем же origin, что и `/api`.

    Включается только при ``API_SERVE_SPA=1`` и наличии собранного ``dist``.
    Существующие файлы отдаются как есть, клиентские маршруты — через
    ``index.html`` (history fallback). Пространство ``/api`` не перекрывается:
    такие запросы получают обычный 404-конверт.
    """
    if config is None or not config.serve_spa:
        return
    if getattr(app.state, "_spa_mounted", False):
        return
    dist = Path(config.spa_dist_dir)
    index = dist / "index.html"
    if not index.is_file():
        return
    app.state._spa_mounted = True
    root = dist.resolve()
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="spa-assets")

    @app.get("/{spa_path:path}", include_in_schema=False)
    async def spa_fallback(spa_path: str) -> FileResponse:
        if spa_path == "api" or spa_path.startswith("api/"):
            raise StarletteHTTPException(status_code=404)
        candidate = (root / spa_path).resolve() if spa_path else root
        if candidate.is_file() and root in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index)


def create_app(
    *,
    config: ApiConfig | None = None,
    services: Services | None = None,
    store: Store | None = None,
) -> FastAPI:
    """Собрать приложение. Аргументы нужны тестам; в production — env."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ensure_app_state(app)
        # Evaluation storage/worker: startup recovery queued/running → interrupted.
        ensure_evaluation_state(app)
        # Env-based config is loaded in ensure_app_state, so the same-origin SPA
        # is mounted here for the `uvicorn src.api.app:app` path as well.
        _mount_spa(app, app.state.config)
        try:
            yield
        finally:
            app.state.hub.shutdown()
            runtime = getattr(app.state, "evaluation", None)
            if runtime is not None:
                runtime.shutdown()

    app = FastAPI(
        title="Structure-Aware Graph RAG API",
        version="1.0.0",
        description="Тонкий adapter над AgentService: исследования, SSE, evidence.",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.services = services
    app.state.store = store
    app.state.hub = None
    app.state.evaluation = None
    app.state._evaluation_lock = None
    app.state._spa_mounted = False

    origins = config.cors_origins if config is not None else DEFAULT_CORS_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Last-Event-ID"],
        max_age=600,
    )

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.payload())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        fields = [
            {
                "loc": [str(item) for item in error.get("loc", ())],
                "msg": str(error.get("msg", "invalid value")),
                "type": str(error.get("type", "value_error")),
            }
            for error in exc.errors()
        ]
        return _error_response(
            422,
            "validation_error",
            _MESSAGES[422],
            details={"fields": fields},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        status = exc.status_code
        if status == 404:
            return _error_response(404, "not_found", _MESSAGES[404])
        if status == 405:
            return _error_response(405, "validation_error", _MESSAGES[405])
        return _error_response(
            status,
            "validation_error" if status < 500 else "internal_error",
            _MESSAGES.get(status, _MESSAGES[500]),
        )

    @app.exception_handler(Exception)
    async def _internal_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        # Без traceback, prompt и текста исключения.
        return _error_response(500, "internal_error", _MESSAGES[500])

    app.include_router(api_router, prefix="/api")
    app.include_router(documents_router, prefix="/api")
    app.include_router(evaluation_router, prefix="/api")
    _mount_spa(app, config)
    return app


app = create_app()
