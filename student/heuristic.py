"""Chấm bàn bằng ô trống, khả năng gộp và thứ tự các ô."""

from functools import lru_cache
from math import log2

from .board import empty_count, empty_positions, is_terminal, spawn, transpose

# Thứ tự trọng số: ô trống, cặp bằng nhau, phạt số mũ, phạt mất thứ tự.
WEIGHTS = {
    "balanced": (135.0, 350.0, 11.0, 47.0),
    "space": (250.0, 450.0, 9.0, 35.0),
    "order": (135.0, 350.0, 11.0, 65.0),
}


# Chỉ tính lại khi gặp một hàng hoặc bộ trọng số mới.
@lru_cache(maxsize=32768)
def line_value(row, variant):
    ranks = []
    occupied = []
    for value in row:
        rank = log2(value) if value else 0.0
        ranks.append(rank)
        if value:
            occupied.append(rank)

    empty = 4 - len(occupied)
    merges = 0
    for index in range(len(occupied) - 1):
        if occupied[index] == occupied[index + 1]:
            merges += 1

    rank_penalty = 0.0
    for rank in ranks:
        rank_penalty += rank**3.5

    increasing = 0.0
    decreasing = 0.0
    for index in range(3):
        difference = ranks[index + 1]**4 - ranks[index]**4
        if difference > 0:
            increasing += difference
        else:
            decreasing -= difference
    disorder = min(increasing, decreasing)

    empty_weight, merge_weight, rank_weight, order_weight = WEIGHTS[variant]
    return (
        empty_weight * empty
        + merge_weight * merges
        - rank_weight * rank_penalty
        - order_weight * disorder
    )


@lru_cache(maxsize=32768)
def line_delta(row, variant):
    """Tổng thay đổi điểm nếu sinh 2 hoặc 4 vào từng chỗ trống trong hàng."""
    baseline = line_value(row, variant)
    total = 0.0
    for index in range(4):
        if row[index] == 0:
            with_two = list(row)
            with_four = list(row)
            with_two[index] = 2
            with_four[index] = 4
            total += (
                0.9 * line_value(tuple(with_two), variant)
                + 0.1 * line_value(tuple(with_four), variant)
                - baseline
            )
    return total


def evaluate(board, variant="space"):
    total = 0.0
    for working in (board, transpose(board)):
        for start in range(0, 16, 4):
            row = working[start:start + 4]
            total += line_value(row, variant)
    return total


def expected_spawn(board, variant="space"):
    count = empty_count(board)
    baseline = 0.0
    correction = 0.0
    for working in (board, transpose(board)):
        for start in range(0, 16, 4):
            row = working[start:start + 4]
            baseline += line_value(row, variant)
            if count:
                correction += line_delta(row, variant)
    if count == 0:
        return baseline
    # Sinh một ô chỉ đổi đúng một hàng và một cột.
    return baseline + correction / count


class Heuristic:
    def __init__(self, variant="space", loss_penalty=1e6):
        if variant not in WEIGHTS:
            raise ValueError("unknown heuristic variant")
        self.variant = variant
        self.loss_penalty = loss_penalty

    def value(self, board):
        value = evaluate(board, self.variant)
        if self.loss_penalty and is_terminal(board):
            value -= self.loss_penalty
        return value

    def expectation(self, board):
        count = empty_count(board)
        if count == 0:
            return self.value(board)
        value = expected_spawn(board, self.variant)
        # Khi chỉ còn một ô trống, sinh ô có thể kết thúc ván ngay.
        if count == 1 and self.loss_penalty:
            position = empty_positions(board)[0]
            if is_terminal(spawn(board, position, 2)):
                value -= 0.9 * self.loss_penalty
            if is_terminal(spawn(board, position, 4)):
                value -= 0.1 * self.loss_penalty
        return value
