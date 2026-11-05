"""Expectimax: chọn hướng tốt nhất, tính trung bình các khả năng sinh ô."""

from time import perf_counter

from .board import chance_outcomes, legal_moves, move


class SearchStopped(Exception):
    """Hết thời gian dành cho một quyết định."""


class Expectimax:
    def __init__(self, heuristic):
        self.heuristic = heuristic
        self.reset()

    def reset(self):
        self.cache = {}
        self.nodes = 0
        self.hits = 0
        self.deadline = None

    def _visit(self):
        self.nodes += 1
        if self.deadline is not None and perf_counter() >= self.deadline:
            raise SearchStopped()

    def root(self, board, depth, directions):
        best_move = directions[0]
        best_value = float("-inf")
        for direction in directions:
            self._visit()
            child, reward, changed = move(board, int(direction))
            if not changed:
                raise ValueError("legal moves disagree with the board")
            value = reward + self.chance(child, depth - 1)
            if value > best_value:
                best_value = value
                best_move = direction
        return best_move, best_value

    def player(self, board, depth):
        self._visit()
        if depth == 0:
            return self.heuristic.value(board)
        key = ("player", depth, board)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        children = legal_moves(board)
        if not children:
            return self.heuristic.value(board)
        best_value = float("-inf")
        for direction, child, reward in children:
            value = reward + self.chance(child, depth - 1)
            if value > best_value:
                best_value = value
        self._remember(key, best_value)
        return best_value

    def chance(self, board, depth):
        self._visit()
        key = ("chance", depth, board)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        if depth == 0:
            # Kỳ vọng sau một lần sinh ô, tính nhanh bằng thay đổi hàng/cột.
            value = self.heuristic.expectation(board)
        else:
            value = 0.0
            for child, probability in chance_outcomes(board):
                value += probability * self.player(child, depth)
        self._remember(key, value)
        return value

    def _remember(self, key, value):
        # Chỉ lưu một nút sau khi đã tính xong tất cả các nhánh của nó.
        if len(self.cache) < 10000:
            self.cache[key] = value
