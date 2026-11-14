"""Тесты детерминированного сравнения структуры отчётов."""

from __future__ import annotations

from src.engine.structure_compare import (
    ADDED,
    EVOLVED,
    EXPANDED,
    REMOVED,
    RENAMED,
    RESTRUCTURED,
    STABLE,
    StructureAnalyzer,
    compare_content_changes,
)


def section(
    title: str,
    *,
    level: int = 1,
    start: int = 1,
    end: int = 2,
    parent: str | None = None,
) -> dict:
    path = f"{parent}/{title}" if parent else title
    return {
        "title": title,
        "section_path": path,
        "breadcrumbs": [parent, title] if parent else [title],
        "level": level,
        "page_start": start,
        "page_end": end,
    }


def snapshots() -> dict[int, list[dict]]:
    return {
        2022: [
            section("Стратегия", start=10, end=12),
            section("Цифровая трансформация", level=2, start=13, end=15),
            section("Управление рисками", start=16, end=20),
            section("ESG", start=21, end=22),
            section("Корпоративное управление", start=23, end=25),
        ],
        2023: [
            section("Стратегия", start=10, end=12),
            section(
                "Цифровая трансформация и технологии",
                level=2,
                start=13,
                end=18,
            ),
            section("Система управления рисками", start=16, end=20),
            section(
                "Корпоративное управление",
                start=23,
                end=25,
                parent="Стратегия",
            ),
            section(
                "Кибербезопасность",
                level=2,
                start=19,
                end=20,
                parent="Управление рисками",
            ),
        ],
    }


def test_detects_added_removed_renamed_and_restructured() -> None:
    analysis = StructureAnalyzer().compare(snapshots(), [2022, 2023])
    changes = analysis["structural_changes"]
    types = {change["change_type"] for change in changes}

    assert RENAMED in types
    assert RESTRUCTURED in types
    assert REMOVED in types
    assert ADDED in types

    removed = [
        change for change in changes if change["change_type"] == REMOVED
    ]
    added = [change for change in changes if change["change_type"] == ADDED]
    assert removed[0]["from_section"]["title"] == "ESG"
    assert added[0]["to_section"]["title"] == "Кибербезопасность"


def test_identical_structures_are_stable() -> None:
    same = {
        2022: snapshots()[2022],
        2023: [dict(item) for item in snapshots()[2022]],
    }
    analysis = StructureAnalyzer().compare(same, [2022, 2023])

    assert analysis["structural_changes"] == []
    assert all(
        item["change_type"] == STABLE for item in analysis["section_matches"]
    )


def test_page_span_detects_expanded_and_reduced() -> None:
    previous = [section("Риски", start=10, end=20)]
    expanded = [section("Риски", start=10, end=30)]
    reduced = [section("Риски", start=10, end=12)]

    expanded_analysis = StructureAnalyzer().compare(
        {2022: previous, 2023: expanded}, [2022, 2023]
    )
    reduced_analysis = StructureAnalyzer().compare(
        {2022: previous, 2023: reduced}, [2022, 2023]
    )

    assert expanded_analysis["structural_changes"][0]["change_type"] == EXPANDED
    assert reduced_analysis["structural_changes"][0]["change_type"] == "reduced"


def test_parent_replaced_by_children_is_split() -> None:
    previous = [section("Риски")]
    current = [
        section("Кредитный риск", parent="Риски", level=2),
        section("Операционный риск", parent="Риски", level=2),
    ]

    analysis = StructureAnalyzer().compare(
        {2022: previous, 2023: current}, [2022, 2023]
    )

    assert any(
        item["change_type"] == RESTRUCTURED
        and item.get("detail") == "section_split"
        for item in analysis["structural_changes"]
    )


def test_children_replaced_by_parent_is_merge() -> None:
    previous = [
        section("Кредитный риск", parent="Риски", level=2),
        section("Операционный риск", parent="Риски", level=2),
    ]
    current = [section("Риски")]

    analysis = StructureAnalyzer().compare(
        {2022: previous, 2023: current}, [2022, 2023]
    )

    assert any(
        item["change_type"] == RESTRUCTURED
        and item.get("detail") == "section_merged"
        for item in analysis["structural_changes"]
    )


def test_content_evolution_with_same_title() -> None:
    matches = [
        {
            "from_year": 2022,
            "to_year": 2023,
            "from_section": {
                "title": "Цифровизация",
                "section_path": "Цифровизация",
            },
            "to_section": {
                "title": "Цифровизация",
                "section_path": "Цифровизация",
            },
            "change_type": STABLE,
        }
    ]
    evidence = {
        "c2022": {
            "year": 2022,
            "section_path": "Цифровизация",
            "text": "Внедрение цифровых сервисов и дистанционных каналов.",
        },
        "c2023": {
            "year": 2023,
            "section_path": "Цифровизация",
            "text": (
                "Технологическая независимость, собственная платформа, "
                "масштабирование и экономический эффект."
            ),
        },
    }

    changes = compare_content_changes(matches, evidence)

    assert changes[0]["change_type"] == EVOLVED
    assert changes[0]["detail"] == "content_changed"


def test_three_years_produce_adjacent_comparisons() -> None:
    analysis = StructureAnalyzer().compare(
        {
            2021: [section("Стратегия")],
            2022: [section("Стратегия")],
            2023: [section("Стратегия", parent="Группа")],
        },
        [2021, 2022, 2023],
    )

    pairs = {
        (item["from_year"], item["to_year"])
        for item in analysis["section_matches"]
    }
    assert pairs == {(2021, 2022), (2022, 2023)}
