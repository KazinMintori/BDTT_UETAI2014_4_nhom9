"""Chấm bàn bằng giá trị ô, điểm gộp và khoảng trống.

Các tiêu chí là quy tắc thiết kế của nhóm, dùng cùng đơn vị giá trị ô.
Mỗi ô trống nhận 2 điểm, bằng giá trị ô nhỏ nhất trong luật 2048.
"""

from functools import lru_cache

from .board import (
    empty_count,
    empty_positions,
    is_terminal,
    merge_row_left,
    spawn,
    transpose,
)


@lru_cache(maxsize=32768)
def line_value(row):
    total_tiles = sum(row)
    empty_points = 2 * row.count(0)
    _, merge_points = merge_row_left(row)

    # Đo mức lệch khỏi thứ tự tăng hoặc giảm bằng chính giá trị các ô.
    increasing = 0
    decreasing = 0
    for index in range(3):
        difference = row[index + 1] - row[index]
        if difference > 0:
            increasing += difference
        else:
            decreasing -= difference
    disorder = min(increasing, decreasing)
    return total_tiles + empty_points + merge_points - disorder


@lru_cache(maxsize=32768)
def line_delta(row):
    """Cộng thay đổi điểm khi sinh 2 hoặc 4 vào mỗi chỗ trống của dòng."""
    baseline = line_value(row)
    total = 0.0
    for index in range(4):
        if row[index] == 0:
            with_two = list(row)
            with_four = list(row)
            with_two[index] = 2
            with_four[index] = 4
            total += (
                0.9 * line_value(tuple(with_two))
                + 0.1 * line_value(tuple(with_four))
                - baseline
            )
    return total


def evaluate(board):
    total = 0.0
    for working in (board, transpose(board)):
        for start in range(0, 16, 4):
            row = working[start:start + 4]
            total += line_value(row)
    return total


def expected_spawn(board):
    count = empty_count(board)
    baseline = 0.0
    correction = 0.0
    for working in (board, transpose(board)):
        for start in range(0, 16, 4):
            row = working[start:start + 4]
            baseline += line_value(row)
            if count:
                correction += line_delta(row)
    if count == 0:
        return baseline
    return baseline + correction / count


class Heuristic:
    def value(self, board):
        # Bàn hết nước nhận 0, các bàn còn chơi được có điểm dương.
        if is_terminal(board):
            return 0.0
        return evaluate(board)

    def expectation(self, board):
        count = empty_count(board)
        if count == 0:
            return self.value(board)
        value = expected_spawn(board)
        if count == 1:
            position = empty_positions(board)[0]
            for tile, probability in ((2, 0.9), (4, 0.1)):
                child = spawn(board, position, tile)
                if is_terminal(child):
                    value -= probability * evaluate(child)
        return value
