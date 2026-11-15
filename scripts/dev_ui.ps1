# Запуск локального демо: FastAPI (8000) + Vite (5173) в отдельных окнах.
# Использование:
#   powershell -ExecutionPolicy Bypass -File scripts\dev_ui.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\dev_ui.ps1 -EvalUi
# Окна остаются открытыми, пока их не закроют; Ctrl+C в каждом окне останавливает сервис.

param(
    [switch]$EvalUi
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$frontend = Join-Path $root "frontend"

if (-not (Test-Path $python)) {
    throw "Не найден $python. Создайте .venv по README."
}
if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    throw "Не найден frontend\node_modules. Выполните 'npm install' в frontend."
}

$apiArgs = @("-m", "uvicorn", "src.api.app:app", "--host", "127.0.0.1", "--port", "8000", "--reload")

if ($EvalUi) {
    $env:EVAL_UI_ENABLED = "1"
}

Start-Process -FilePath $python -ArgumentList $apiArgs -WorkingDirectory $root -WindowStyle Normal
Start-Process -FilePath "npm.cmd" -ArgumentList @("run", "dev") -WorkingDirectory $frontend -WindowStyle Normal

Write-Host ""
Write-Host "Backend: http://127.0.0.1:8000/api/health"
Write-Host "UI:      http://localhost:5173"
if ($EvalUi) {
    Write-Host "Evaluation: http://localhost:5173/evaluation (EVAL_UI_ENABLED=1)"
}
Write-Host ""
Write-Host "Остановка: Ctrl+C в окнах backend/frontend."
