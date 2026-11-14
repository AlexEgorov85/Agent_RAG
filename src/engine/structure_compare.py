"""Детерминированное сопоставление деревьев разделов между отчётами.

Первый проход — нормализованные названия, иерархия и близость родителей.
Второй (опциональный) — LLM проверяет только спорные пары, чтобы не отдавать
ей сотни очевидных заголовков.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any

_TITLE_CLEAN = re.compile(r"[^\wа-яё]+", re.IGNORECASE)
_STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "по",
    "с",
    "со",
    "для",
    "о",
    "об",
    "от",
    "до",
    "из",
    "к",
    "ко",
    "the",
    "and",
    "of",
    "to",
    "for",
}

STABLE = "stable"
ADDED = "added"
REMOVED = "removed"
RENAMED = "renamed"
RESTRUCTURED = "restructured"
EXPANDED = "expanded"
REDUCED = "reduced"
EVOLVED = "evolved"

_ALLOWED_CHANGES = {
    STABLE,
    ADDED,
    REMOVED,
    RENAMED,
    RESTRUCTURED,
    EXPANDED,
    REDUCED,
    EVOLVED,
}


def _normalize(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return _TITLE_CLEAN.sub(" ", value.casefold().replace("ё", "е")).strip()


def _normalize_path(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return "/".join(
        segment
        for segment in (
            _normalize(part) for part in value.strip("/").split("/")
        )
        if segment
    )


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in _normalize(value).split()
        if len(token) > 1 and token not in _STOPWORDS
    }


def _title_similarity(left: Any, right: Any) -> float:
    left_text = _normalize(left)
    right_text = _normalize(right)
    if not left_text or not right_text:
        return 0.0
    if left_text == right_text:
        return 1.0
    sequence = SequenceMatcher(None, left_text, right_text).ratio()
    left_tokens, right_tokens = _tokens(left_text), _tokens(right_text)
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    return 0.55 * sequence + 0.45 * jaccard


def _breadcrumbs(section: Mapping[str, Any]) -> list[str]:
    value = section.get("breadcrumbs")
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    path = section.get("section_path")
    if isinstance(path, str):
        return [part.strip() for part in path.split("/") if part.strip()]
    title = section.get("title")
    return [str(title)] if title else []


def _section_path(section: Mapping[str, Any]) -> str:
    return "/".join(_breadcrumbs(section))


def _parent_path(section: Mapping[str, Any]) -> str:
    crumbs = _breadcrumbs(section)
    return "/".join(crumbs[:-1])


def _parent_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_parent = _normalize(_parent_path(left))
    right_parent = _normalize(_parent_path(right))
    if not left_parent and not right_parent:
        return 1.0
    if left_parent == right_parent:
        return 1.0
    if left_parent and right_parent and (
        left_parent.startswith(right_parent)
        or right_parent.startswith(left_parent)
    ):
        return 0.75
    return SequenceMatcher(None, left_parent, right_parent).ratio() * 0.5


def _page_span(section: Mapping[str, Any]) -> int | None:
    start = section.get("page_start")
    end = section.get("page_end")
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        return end - start + 1
    return None


def _level(section: Mapping[str, Any]) -> int | None:
    value = section.get("level")
    return value if isinstance(value, int) else None


def _pair_score(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    title = _title_similarity(left.get("title"), right.get("title"))
    parent = _parent_similarity(left, right)
    left_level, right_level = _level(left), _level(right)
    level = 1.0 if left_level == right_level else 0.5
    return 0.68 * title + 0.22 * parent + 0.10 * level


def _section_summary(section: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "title": section.get("title"),
        "section_path": _section_path(section),
        "level": _level(section),
        "page_start": section.get("page_start"),
        "page_end": section.get("page_end"),
    }


def _section_text(
    evidence: Mapping[str, Mapping[str, Any]],
    year: int,
    path: str | None,
) -> str:
    if not path:
        return ""
    normalized_path = _normalize_path(path)
    parts: list[str] = []
    for chunk in evidence.values():
        if not isinstance(chunk, Mapping):
            continue
        try:
            chunk_year = int(chunk.get("year"))
        except (TypeError, ValueError):
            continue
        if chunk_year != year:
            continue
        chunk_path = _normalize_path(chunk.get("section_path"))
        if not chunk_path:
            continue
        if not (
            chunk_path == normalized_path
            or chunk_path.startswith(normalized_path + "/")
        ):
            continue
        parts.append(
            str(chunk.get("text") or "")
            + " "
            + str(chunk.get("html") or "")
        )
    return " ".join(part for part in parts if part.strip())


def classify_content_change(
    previous_text: str,
    current_text: str,
) -> tuple[str, float, float] | None:
    """Сравнивает содержание раздела при неизменной структуре."""
    if not previous_text.strip() or not current_text.strip():
        return None
    previous_tokens, current_tokens = _tokens(previous_text), _tokens(current_text)
    union = previous_tokens | current_tokens
    similarity = (
        len(previous_tokens & current_tokens) / len(union) if union else 1.0
    )
    length_ratio = len(current_text) / max(1, len(previous_text))
    if similarity < 0.45:
        change_type = EVOLVED
    elif length_ratio > 1.35:
        change_type = EXPANDED
    elif length_ratio < 0.65:
        change_type = REDUCED
    else:
        change_type = STABLE
    return change_type, similarity, length_ratio


def compare_content_changes(
    matches: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Ищет изменения содержания по уже найденным chunk'ам разделов."""
    changes: list[dict[str, Any]] = []
    for match in matches:
        if not isinstance(match, Mapping):
            continue
        from_section = match.get("from_section")
        to_section = match.get("to_section")
        from_year = match.get("from_year")
        to_year = match.get("to_year")
        if not isinstance(from_year, int) or not isinstance(to_year, int):
            continue
        if not isinstance(from_section, Mapping) or not isinstance(
            to_section, Mapping
        ):
            continue
        previous_text = _section_text(
            evidence, from_year, from_section.get("section_path")
        )
        current_text = _section_text(
            evidence, to_year, to_section.get("section_path")
        )
        classified = classify_content_change(previous_text, current_text)
        if classified is None:
            continue
        change_type, similarity, length_ratio = classified
        if change_type == STABLE:
            continue
        changes.append(
            {
                "from_year": from_year,
                "to_year": to_year,
                "from_section": dict(from_section),
                "to_section": dict(to_section),
                "score": round(similarity, 3),
                "change_type": change_type,
                "detail": "content_changed",
                "content_similarity": round(similarity, 3),
                "length_ratio": round(length_ratio, 3),
            }
        )
    return changes


class StructureAnalyzer:
    """Сопоставляет snapshots разделов и классифицирует изменения."""

    def __init__(
        self,
        client: Any = None,
        model: str | None = None,
        *,
        match_threshold: float = 0.50,
        ambiguous_threshold: float = 0.32,
        verify_threshold: float = 0.72,
        max_ambiguous: int = 8,
        max_tokens: int = 512,
    ) -> None:
        self.client = client
        self.model = model
        self.match_threshold = match_threshold
        self.ambiguous_threshold = ambiguous_threshold
        self.verify_threshold = verify_threshold
        self.max_ambiguous = max_ambiguous
        self.max_tokens = max_tokens

    def compare(
        self,
        snapshots: Mapping[int, Sequence[Mapping[str, Any]]],
        years: Sequence[int] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        ordered = sorted(years or snapshots)
        ordered = [year for year in ordered if year in snapshots]
        matches: list[dict[str, Any]] = []
        changes: list[dict[str, Any]] = []

        for previous_year, current_year in zip(ordered, ordered[1:]):
            previous = list(snapshots.get(previous_year, []))
            current = list(snapshots.get(current_year, []))
            pair_matches, unmatched_previous, unmatched_current = self._match_pair(
                previous, current
            )
            pair_changes: list[dict[str, Any]] = []
            for left, right, score, change_type, detail in pair_matches:
                match = {
                    "from_year": previous_year,
                    "to_year": current_year,
                    "from_section": _section_summary(left),
                    "to_section": _section_summary(right),
                    "score": round(score, 3),
                    "change_type": change_type,
                    "detail": detail,
                }
                matches.append(match)
                if change_type != STABLE:
                    pair_changes.append(dict(match))

            for section in unmatched_previous:
                pair_changes.append(
                    {
                        "from_year": previous_year,
                        "to_year": current_year,
                        "from_section": _section_summary(section),
                        "to_section": None,
                        "score": 0.0,
                        "change_type": REMOVED,
                        "detail": "section_not_found_in_later_report",
                    }
                )
            for section in unmatched_current:
                pair_changes.append(
                    {
                        "from_year": previous_year,
                        "to_year": current_year,
                        "from_section": None,
                        "to_section": _section_summary(section),
                        "score": 0.0,
                        "change_type": ADDED,
                        "detail": "section_not_found_in_earlier_report",
                    }
                )
            changes.extend(
                self._collapse_split_merge(pair_changes)
            )

        return {"section_matches": matches, "structural_changes": changes}

    @staticmethod
    def _collapse_split_merge(
        changes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Превращает «удалён родитель + добавлены дети» в split/merge."""
        removed = [
            item for item in changes if item.get("change_type") == REMOVED
        ]
        added = [item for item in changes if item.get("change_type") == ADDED]
        consumed: set[int] = set()
        collapsed: list[dict[str, Any]] = []

        for parent in removed:
            section = parent.get("from_section")
            if not isinstance(section, Mapping):
                continue
            path = _normalize_path(section.get("section_path"))
            if not path:
                continue
            children = [
                child
                for child in added
                if id(child) not in consumed
                and _normalize_path(
                    (child.get("to_section") or {}).get("section_path")
                ).startswith(path + "/")
            ]
            if len(children) < 2:
                continue
            consumed.add(id(parent))
            consumed.update(id(child) for child in children)
            collapsed.append(
                {
                    "from_year": parent.get("from_year"),
                    "to_year": parent.get("to_year"),
                    "from_section": dict(section),
                    "to_section": None,
                    "score": 0.0,
                    "change_type": RESTRUCTURED,
                    "detail": "section_split",
                    "related_sections": [
                        (child.get("to_section") or {}).get("title")
                        for child in children
                    ],
                }
            )

        for parent in added:
            if id(parent) in consumed:
                continue
            section = parent.get("to_section")
            if not isinstance(section, Mapping):
                continue
            path = _normalize_path(section.get("section_path"))
            if not path:
                continue
            children = [
                child
                for child in removed
                if id(child) not in consumed
                and _normalize_path(
                    (child.get("from_section") or {}).get("section_path")
                ).startswith(path + "/")
            ]
            if len(children) < 2:
                continue
            consumed.add(id(parent))
            consumed.update(id(child) for child in children)
            collapsed.append(
                {
                    "from_year": parent.get("from_year"),
                    "to_year": parent.get("to_year"),
                    "from_section": None,
                    "to_section": dict(section),
                    "score": 0.0,
                    "change_type": RESTRUCTURED,
                    "detail": "section_merged",
                    "related_sections": [
                        (child.get("from_section") or {}).get("title")
                        for child in children
                    ],
                }
            )

        return [
            item for item in changes if id(item) not in consumed
        ] + collapsed

    def _match_pair(
        self,
        previous: list[Mapping[str, Any]],
        current: list[Mapping[str, Any]],
    ) -> tuple[
        list[tuple[Mapping[str, Any], Mapping[str, Any], float, str, str]],
        list[Mapping[str, Any]],
        list[Mapping[str, Any]],
    ]:
        candidates: list[tuple[float, int, int]] = []
        for left_index, left in enumerate(previous):
            for right_index, right in enumerate(current):
                score = _pair_score(left, right)
                if score >= self.ambiguous_threshold:
                    candidates.append((score, left_index, right_index))
        candidates.sort(reverse=True)

        used_left: set[int] = set()
        used_right: set[int] = set()
        pairs: list[tuple[Mapping[str, Any], Mapping[str, Any], float, str, str]] = []
        ambiguous: list[tuple[float, int, int]] = []
        verified_pairs = 0

        for score, left_index, right_index in candidates:
            if left_index in used_left or right_index in used_right:
                continue
            if score < self.match_threshold:
                ambiguous.append((score, left_index, right_index))
                continue
            left, right = previous[left_index], current[right_index]
            change_type, detail = self._classify(left, right, score)
            if (
                self.client is not None
                and score < self.verify_threshold
                and verified_pairs < self.max_ambiguous
            ):
                verified_pairs += 1
                verified = self._verify_pair(left, right, score)
                if verified is False:
                    # LLM считает пары разными: оба раздела идут в add/remove.
                    continue
                if isinstance(verified, str) and verified in _ALLOWED_CHANGES:
                    change_type = verified
                    detail = "verified_by_llm"
            used_left.add(left_index)
            used_right.add(right_index)
            pairs.append((left, right, score, change_type, detail))

        # Спорные пары: если LLM доступна, она решает; иначе они остаются
        # добавленными/удалёнными, а не молча склеиваются.
        for score, left_index, right_index in ambiguous:
            if left_index in used_left or right_index in used_right:
                continue
            if self.client is None or verified_pairs >= self.max_ambiguous:
                continue
            left, right = previous[left_index], current[right_index]
            verified = self._verify_pair(left, right, score)
            verified_pairs += 1
            if verified is False:
                continue
            change_type = (
                verified
                if isinstance(verified, str) and verified in _ALLOWED_CHANGES
                else self._classify(left, right, score)[0]
            )
            used_left.add(left_index)
            used_right.add(right_index)
            pairs.append((left, right, score, change_type, "verified_by_llm"))

        unmatched_previous = [
            section for index, section in enumerate(previous) if index not in used_left
        ]
        unmatched_current = [
            section for index, section in enumerate(current) if index not in used_right
        ]
        return pairs, unmatched_previous, unmatched_current

    def _classify(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
        score: float,
    ) -> tuple[str, str]:
        del score
        title_changed = _normalize(left.get("title")) != _normalize(right.get("title"))
        parent_changed = _normalize(_parent_path(left)) != _normalize(
            _parent_path(right)
        )
        level_changed = _level(left) != _level(right)
        left_span, right_span = _page_span(left), _page_span(right)

        if title_changed and parent_changed:
            return RESTRUCTURED, "renamed_and_moved"
        if title_changed:
            return RENAMED, "title_changed"
        if parent_changed or level_changed:
            return RESTRUCTURED, "moved_or_level_changed"
        if left_span is not None and right_span is not None:
            difference = right_span - left_span
            if difference >= max(2, left_span * 0.25):
                return EXPANDED, "page_span_grew"
            if difference <= -max(2, left_span * 0.25):
                return REDUCED, "page_span_shrank"
        return STABLE, "same_title_and_parent"

    def _verify_pair(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
        score: float,
    ) -> bool | str | None:
        if self.client is None or not self.model:
            return None
        payload = {
            "left": {
                "title": left.get("title"),
                "breadcrumbs": _breadcrumbs(left),
                "level": _level(left),
                "pages": [left.get("page_start"), left.get("page_end")],
            },
            "right": {
                "title": right.get("title"),
                "breadcrumbs": _breadcrumbs(right),
                "level": _level(right),
                "pages": [right.get("page_start"), right.get("page_end")],
            },
            "deterministic_score": round(score, 3),
        }
        instruction = (
            "Ты сопоставляешь разделы годовых отчётов. Верни только JSON: "
            '{"same_section": true|false, "change_type": '
            '"stable|renamed|restructured|expanded|reduced", "reason": "..."}. '
            "same_section=true, только если это один и тот же тематический раздел, "
            "даже если название изменилось. Не используй внешние знания."
        )
        try:
            completions = self.client.chat.completions
            create = getattr(completions, "create", completions)
            response = create(
                model=self.model,
                messages=[
                    {"role": "system", "content": instruction},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            decoded = json.loads(str(content))
            if decoded.get("same_section") is False:
                return False
            change_type = decoded.get("change_type")
            return change_type if change_type in _ALLOWED_CHANGES else True
        except Exception:
            return None
