"""Поведение ленивого AgentProvider при временном сбое старта."""

from __future__ import annotations

import pytest

from src.api.errors import BackendUnavailable
from src.api.services import AgentProvider


def test_failed_start_is_cached_during_retry_window() -> None:
    calls = 0

    def factory() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("temporary")

    provider = AgentProvider(factory, retry_after_s=60)
    with pytest.raises(BackendUnavailable):
        provider.get()
    with pytest.raises(BackendUnavailable):
        provider.get()

    assert calls == 1
    assert provider.available() is False


def test_failed_start_retries_after_window() -> None:
    calls = 0

    def factory() -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary")
        return object()

    provider = AgentProvider(factory, retry_after_s=0)
    with pytest.raises(BackendUnavailable):
        provider.get()

    assert provider.get() is not None
    assert calls == 2


def test_set_service_resets_failed_state() -> None:
    provider = AgentProvider(lambda: (_ for _ in ()).throw(RuntimeError()), retry_after_s=60)
    with pytest.raises(BackendUnavailable):
        provider.get()

    marker = object()
    provider.set_service(marker)

    assert provider.get() is marker
