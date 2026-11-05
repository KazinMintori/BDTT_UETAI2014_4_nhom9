"""Agent chơi 2048 sử dụng Expectimax, Iterative Deepening và phân bổ thời gian động."""

from time import perf_counter

from game2048.observation import Observation
from game2048.types import Move

from .board import empty_count, from_matrix, move
from .heuristic import Heuristic
from .search import Expectimax, SearchStopped


class Agent:
    name = "expectimax-nhom9"

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.heuristic = Heuristic()
        self.search = Expectimax(self.heuristic)
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        """Khởi tạo lại trạng thái của Agent trước mỗi ván chơi mới."""
        self.search.reset()

    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        """Đưa ra quyết định nước đi tối ưu dựa trên thời gian cho phép."""
        started = perf_counter()
        if not obs.legal_moves:
            raise ValueError("Bàn cờ đã kết thúc (terminal board)")

        board = from_matrix(obs.board)
        legal_moves = tuple(sorted(obs.legal_moves))

        # 1. Nước đi dự phòng 1-ply (Fallback move) để đảm bảo luôn có nước đi hợp lệ
        best_move = self._fallback_move(board, legal_moves)

        # Nếu chỉ có 1 nước đi duy nhất, không cần tốn thời gian suy nghĩ
        if len(legal_moves) == 1:
            return best_move

        # 2. Phân bổ ngân sách thời gian cho nước đi hiện tại
        budget_ms = self._calculate_budget(obs, time_limit_ms, board)
        if budget_ms < 0.5:
            return best_move

        # Đặt deadline với một khoảng dự phòng an toàn (90% thời gian tính toán)
        usable_ms = budget_ms * 0.90
        deadline = started + usable_ms / 1000.0
        self.search.reset()
        self.search.configure_deadline(deadline)

        # 3. Tìm kiếm lặp sâu dần (Iterative Deepening Search)
        for depth in range(1, self.max_depth + 1):
            try:
                candidate, _ = self.search.root_search(board, depth, legal_moves)
                # Chỉ cập nhật nước đi khi đã hoàn thành trọn vẹn độ sâu này
                best_move = candidate
            except SearchStopped:
                # Khi hết giờ, dừng vòng lặp an toàn và trả về kết quả độ sâu trước đó
                break

        return best_move

    def _fallback_move(self, board: tuple, legal_moves: tuple) -> Move:
        """Tìm nhanh nước đi tốt nhất theo heuristic 1-ply để dự phòng."""
        best_move = legal_moves[0]
        best_score = float("-inf")

        for direction in legal_moves:
            next_board, reward, changed = move(board, int(direction))
            if changed:
                score = reward + self.heuristic.evaluate(next_board)
                if score > best_score:
                    best_score = score
                    best_move = direction

        return best_move

    def _calculate_budget(self, obs: Observation, time_limit_ms: float, board: tuple) -> float:
        """Tính toán thời gian suy nghĩ hợp lý cho nước đi hiện tại.
        
        Chiến lược:
        - Lấy thời gian còn lại chia đều cho số nước đi dự kiến (khoảng 1.000 nước).
        - Điều chỉnh: Bàn cờ thoáng (>= 7 ô) đi nhanh 0.6x; bàn cờ nguy cấp (<= 3 ô) suy nghĩ kỹ 1.6x.
        """
        time_left_ms = obs.time_remaining_ms if obs.time_remaining_ms is not None else time_limit_ms
        if time_left_ms is None or time_left_ms <= 3.0:
            return 0.0

        estimated_moves = max(200.0, 1000.0 - obs.move_count)
        base_budget = time_left_ms / estimated_moves

        empty = empty_count(board)
        if empty <= 3:
            multiplier = 1.6
        elif empty >= 7:
            multiplier = 0.6
        else:
            multiplier = 1.0

        budget = base_budget * multiplier
        return min(budget, 60.0, max(0.0, time_left_ms - 2.0))
