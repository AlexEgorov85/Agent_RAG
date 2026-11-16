"""Текстовые утилиты, общие для retrieval-слоя."""

from __future__ import annotations


def bounded_edit_distance(left: str, right: str) -> int:
    """Расстояние Левенштейна с ранним выходом для разницы длин больше 1.

    Значения больше 1 не различимы: функция предназначена только для
    исправления опечаток в один символ.
    """
    if left == right:
        return 0
    if abs(len(left) - len(right)) > 1:
        return 2
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]
