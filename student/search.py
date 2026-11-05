"""Thuật toán tìm kiếm Expectimax chuẩn cho game 2048.

Mô hình hóa:
- Lượt người chơi (Max node): Chọn nước đi có giá trị kỳ vọng lớn nhất.
- Lượt môi trường (Chance node): Tính trung bình kỳ vọng sinh ô 2 (90%) và ô 4 (10%).
- Sử dụng Transposition Table để lưu nhớ các trạng thái đã tính.
"""

from time import perf_counter

from .board import chance_outcomes, legal_moves, move


class SearchStopped(Exception):
    """Ngoại lệ an toàn khi hết thời gian suy nghĩ của nước đi."""


class Expectimax:
    def __init__(self, heuristic, max_cache: int = 50_000):
        self.heuristic = heuristic
        self.max_cache = max_cache
        self.reset()

    def reset(self) -> None:
        """Xóa bộ nhớ đệm (cache) trước mỗi ván hoặc mỗi lượt tìm kiếm."""
        self.cache = {}
        self.nodes = 0
        self.hits = 0
        self.deadline = None

    def configure_deadline(self, deadline: float) -> None:
        """Thiết lập mốc thời gian giới hạn (deadline)."""
        self.deadline = deadline

    def root_search(self, board: tuple, depth: int, directions: tuple):
        """Tìm kiếm tại nút gốc (giai đoạn đầu của mỗi độ sâu trong IDS)."""
        best_move = directions[0]
        best_value = float("-inf")

        for direction in directions:
            self._check_time()
            next_board, reward, changed = move(board, int(direction))
            if not changed:
                continue

            # Giá trị = Điểm gộp ngay lập tức + Kỳ vọng từ lượt sinh ô của môi trường
            value = reward + self.chance_node(next_board, depth - 1)
            if value > best_value:
                best_value = value
                best_move = direction

        return best_move, best_value

    def max_node(self, board: tuple, depth: int) -> float:
        """Nút người chơi: Thử tất cả các nước đi hợp lệ và lấy giá trị lớn nhất."""
        self._check_time()

        if depth <= 0:
            return self.heuristic.evaluate(board)

        # Kiểm tra bảng nhớ tạm (Transposition Table)
        key = (board, depth)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        children = legal_moves(board)
        if not children:
            return -1e6  # Bàn cờ không còn nước đi (Game Over)

        best_value = float("-inf")
        for direction, next_board, reward in children:
            value = reward + self.chance_node(next_board, depth - 1)
            if value > best_value:
                best_value = value

        if len(self.cache) < self.max_cache:
            self.cache[key] = best_value
        return best_value

    def chance_node(self, board: tuple, depth: int) -> float:
        """Nút môi trường: Tính kỳ vọng toán học khi sinh ô 2 (90%) và ô 4 (10%)."""
        self._check_time()

        if depth <= 0:
            return self.heuristic.evaluate(board)

        positions = empty_positions(board)
        if not positions:
            return self.max_node(board, depth)

        # Tối ưu nhẹ: Khi có quá nhiều ô trống ở độ sâu lớn, chỉ xét 4 ô trống đầu tiên
        # để tránh bùng nổ tổ hợp nhánh và giữ tốc độ ổn định cho Python
        if len(positions) > 4 and depth > 1:
            positions = positions[:4]

        count = len(positions)
        total = 0.0

        for pos in positions:
            # 90% cơ hội sinh ô 2
            board_2 = spawn(board, pos, 2)
            total += 0.9 * self.max_node(board_2, depth - 1)

            # 10% cơ hội sinh ô 4
            board_4 = spawn(board, pos, 4)
            total += 0.1 * self.max_node(board_4, depth - 1)

        return total / count

    def _check_time(self) -> None:
        """Kiểm tra thời gian sau mỗi 64 nút để giảm tải chi phí gọi hàm đồng hồ."""
        self.nodes += 1
        if self.nodes % 64 == 0:
            if self.deadline is not None and perf_counter() >= self.deadline:
                raise SearchStopped("Hết thời gian cho phép")
