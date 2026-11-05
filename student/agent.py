"""Agent nhìn trước tối đa hai nước, dùng bàn tuple và vòng lặp thông thường."""

from time import perf_counter

from .board import from_matrix, move
from .heuristic import Heuristic
from .search import Expectimax, SearchStopped


class Agent:
    name = "expectimax-simple"

    def __init__(self, search_depth=2, time_per_move_ms=10):
        if search_depth not in (1, 2):
            raise ValueError("search_depth must be 1 or 2")
        if time_per_move_ms <= 0:
            raise ValueError("time_per_move_ms must be positive")
        self.search_depth = search_depth
        # Mốc 10 ms là cấu hình thời gian, có thể thay khi chạy thử.
        # Với ngân sách 60 giây, mức này dành chỗ cho khoảng 6000 quyết định.
        self.time_per_move_ms = time_per_move_ms
        self.heuristic = Heuristic()
        self.search = Expectimax(self.heuristic)
        self.reset()

    def reset(self, seed=None):
        self.search.reset()
        self.last_nodes = 0
        self.last_completed_depth = 0

    def choose_move(self, obs, time_limit_ms):
        started = perf_counter()
        if not obs.legal_moves:
            raise ValueError("choose_move called on a terminal board")
        board = from_matrix(obs.board)
        directions = tuple(sorted(obs.legal_moves))
        self.search.reset()
        self.last_nodes = 0
        self.last_completed_depth = 0

        # Chuẩn bị một nước hợp lệ trước khi bắt đầu tìm kiếm.
        best_move = self._fallback_move(board, directions)
        if len(directions) == 1:
            return best_move

        available_ms = max(0.0, float(time_limit_ms))
        if obs.time_remaining_ms is not None:
            available_ms = min(available_ms, max(0.0, obs.time_remaining_ms))
        # Giữ 20% thời gian được cấp để trả kết quả và xử lý sai số đồng hồ.
        budget_ms = min(self.time_per_move_ms, available_ms * 0.8)
        if budget_ms < 0.5:
            return best_move
        self.search.deadline = started + budget_ms / 1000.0

        # Depth 1: đi một nước, sinh ô, chấm bàn.
        # Depth 2: sau khi sinh ô còn nhìn thêm một lượt đi nữa.
        for depth in range(1, self.search_depth + 1):
            try:
                candidate, value = self.search.root(board, depth, directions)
            except SearchStopped:
                break
            best_move = candidate
            self.last_completed_depth = depth
        self.last_nodes = self.search.nodes
        return best_move

    def _fallback_move(self, board, directions):
        best_move = directions[0]
        best_value = float("-inf")
        for direction in directions:
            child, reward, changed = move(board, int(direction))
            if not changed:
                raise ValueError("legal moves disagree with the board")
            value = reward + self.heuristic.value(child)
            if value > best_value:
                best_value = value
                best_move = direction
        return best_move
