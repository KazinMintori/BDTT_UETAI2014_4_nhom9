"""Bàn cờ là tuple 16 giá trị, đọc theo từng hàng từ trái sang phải.

Ví dụ hàng đầu [2, 0, 4, 4] được giữ nguyên giá trị 2, 0, 4, 4.
Tuple giúp dùng bàn làm khóa cache và tránh sửa nhầm bàn của nhánh khác.
"""

from functools import lru_cache

MAX_TILE = 2**31


def from_matrix(matrix):
    if matrix.shape != (4, 4):
        raise ValueError("board must have shape (4, 4)")
    values = []
    for cell in matrix.flat:
        value = int(cell)
        if value != 0:
            if value < 2:
                raise ValueError("tiles must be zero or powers of two starting at 2")
            if value > MAX_TILE:
                raise OverflowError("tile exceeds engine uint32 range")
            # Chia liên tiếp cho 2 để kiểm tra giá trị ô hợp lệ.
            remaining = value
            while remaining > 1 and remaining % 2 == 0:
                remaining //= 2
            if remaining != 1:
                raise ValueError("tiles must be zero or powers of two")
        values.append(value)
    return tuple(values)


def to_matrix(board):
    import numpy as np

    return np.array(board, dtype=np.uint32).reshape(4, 4)


def transpose(board):
    result = []
    for column in range(4):
        for row in range(4):
            result.append(board[row * 4 + column])
    return tuple(result)


def empty_positions(board):
    positions = []
    for index in range(16):
        if board[index] == 0:
            positions.append(index)
    return positions


def empty_count(board):
    return board.count(0)


def spawn(board, position, tile):
    if position < 0 or position >= 16 or board[position] != 0:
        raise ValueError("spawn requires an empty cell")
    if tile not in (2, 4):
        raise ValueError("new tile must be 2 or 4")
    result = list(board)
    result[position] = tile
    return tuple(result)


# Cùng một hàng luôn cho cùng kết quả gộp. Python nhớ tối đa 32768 hàng.
@lru_cache(maxsize=32768)
def merge_row_left(row):
    occupied = []
    for value in row:
        if value != 0:
            occupied.append(value)

    result = []
    reward = 0
    index = 0
    while index < len(occupied):
        value = occupied[index]
        if index + 1 < len(occupied) and value == occupied[index + 1]:
            value *= 2
            if value > MAX_TILE:
                raise OverflowError("tile 2**32 exceeds engine uint32")
            reward += value
            index += 2  # Bỏ cả cặp, ô vừa tạo không gộp lần nữa.
        else:
            index += 1
        result.append(value)

    while len(result) < 4:
        result.append(0)
    return tuple(result), reward


def move(board, direction):
    if direction not in (0, 1, 2, 3):
        raise ValueError("direction must be UP=0 DOWN=1 LEFT=2 RIGHT=3")

    # Chuyển vị để nước dọc cũng xử lý bằng bốn hàng ngang.
    vertical = direction in (0, 1)
    reverse = direction in (1, 3)
    working = transpose(board) if vertical else board
    result = []
    reward = 0
    for start in range(0, 16, 4):
        row = working[start:start + 4]
        if reverse:
            row = tuple(reversed(row))
        merged, points = merge_row_left(row)
        if reverse:
            merged = tuple(reversed(merged))
        result.extend(merged)
        reward += points

    result = tuple(result)
    if vertical:
        result = transpose(result)
    return result, reward, result != board


@lru_cache(maxsize=8192)
def legal_moves(board):
    children = []
    for direction in range(4):
        child, reward, changed = move(board, direction)
        if changed:
            children.append((direction, child, reward))
    return tuple(children)


def is_terminal(board):
    if not any(board):
        return True
    if 0 in board:
        return False
    for row in range(4):
        for column in range(4):
            index = row * 4 + column
            if column < 3 and board[index] == board[index + 1]:
                return False
            if row < 3 and board[index] == board[index + 4]:
                return False
    return True


def chance_outcomes(board):
    positions = empty_positions(board)
    if not positions:
        yield board, 1.0
        return
    count = len(positions)
    for position in positions:
        yield spawn(board, position, 2), 0.9 / count
        yield spawn(board, position, 4), 0.1 / count
