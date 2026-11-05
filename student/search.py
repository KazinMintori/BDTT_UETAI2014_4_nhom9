"""Expectimax: lượt người chơi lấy max, lượt sinh ô lấy kỳ vọng."""

from time import perf_counter

from .board import empty_positions, legal_moves, move, spawn


class SearchStopped(Exception):
    """Dừng do hết thời gian hoặc số nút cho phép; không che lỗi lập trình."""


class Expectimax:
    def __init__(
        self,
        heuristic,
        cutoff=0.003,
        reward_weight=0.1,
        cache_enabled=True,
        max_cache=50_000,
        loss_penalty=1e9,
    ):
        self.heuristic = heuristic
        self.evaluator = heuristic.value
        self.leaf_expectation = heuristic.expectation
        self.cutoff = cutoff
        self.reward_weight = reward_weight
        self.cache_enabled = cache_enabled
        self.max_cache = max_cache
        self.loss_penalty = loss_penalty
        self.reset()

    def reset(self) -> None:
        self.cache = {}
        # Cache vẫn chỉ chứa số; tập riêng ghi cây con từng chạm giới hạn độ sâu.
        self.depth_limited_cache = set()
        self.horizon_visits = 0
        self.probability_saturated = False
        self.nodes = self.hits = self.lookups = 0
        self.deadline = self.node_limit = None

    def configure_limits(self, deadline=None, node_limit=None) -> None:
        self.deadline = deadline
        self.node_limit = node_limit

    def root(self, board: tuple, depth: int, directions):
        """Depth 1 là một nước đi, một lần sinh ô, rồi đánh giá bàn cờ."""
        self.horizon_visits = 0
        self.probability_saturated = False
        best_move, best_value = directions[0], float("-inf")
        for direction in directions:
            if self.deadline is not None and perf_counter() >= self.deadline:
                raise SearchStopped("root deadline")
            next_board, reward, changed = move(board, int(direction))
            if not changed:
                raise ValueError("root legal moves disagree with simulator")
            value = self.reward_weight * reward + self.chance(next_board, depth - 1, 1.0)
            if value > best_value:
                best_move, best_value = direction, value
        # Nếu không lá nào dừng vì độ sâu, tăng depth không thể mở rộng thêm
        # cây với cutoff hiện tại. Điều này không chứng minh nước đi tối ưu.
        self.probability_saturated = self.horizon_visits == 0
        return best_move, best_value

    def player(self, board: tuple, depth: int, probability=1.0) -> float:
        self._visit()
        if depth == 0 or probability < self.cutoff:
            if depth == 0 and probability >= self.cutoff:
                self.horizon_visits += 1
            return self.evaluator(board)

        # Cùng bàn cờ nhưng xác suất đường đi khác có thể dừng ở độ sâu khác.
        context = probability if self.cutoff else 1.0
        key = (0, depth, board, context)
        if self.cache_enabled:
            self.lookups += 1
            cached = self.cache.get(key)
            if cached is not None:
                self.hits += 1
                if key in self.depth_limited_cache:
                    self.horizon_visits += 1
                return cached

        horizon_start = self.horizon_visits
        children = legal_moves(board)
        if not children:
            value = self.evaluator(board)
            if not self.heuristic.loss_penalty:
                value -= self.loss_penalty
            return self._remember(key, value, horizon_start)

        best_value = float("-inf")
        next_depth = depth - 1
        for _, next_board, reward in children:
            value = self.reward_weight * reward + self.chance(next_board, next_depth, probability)
            if value > best_value:
                best_value = value
        return self._remember(key, best_value, horizon_start)

    def chance(self, board: tuple, depth: int, probability=1.0) -> float:
        self._visit()
        positions = empty_positions(board)
        count = len(positions)
        if not count:
            return self.player(board, depth, probability)

        probability_2 = probability * 0.9 / count
        is_leaf = depth == 0 or probability_2 < self.cutoff
        # Khóa lá dùng chung giữa các depth và xác suất. Lý do dừng phải xét
        # lại ngay cả khi giá trị lá được lấy từ cache.
        if depth == 0 and probability_2 >= self.cutoff:
            self.horizon_visits += 1
        context = probability if self.cutoff else 1.0
        # Khi cả hai nhánh đều dừng, kỳ vọng lá không còn phụ thuộc depth và p.
        key = (1, 0, board, 0.0) if is_leaf else (1, depth, board, context)
        if self.cache_enabled:
            self.lookups += 1
            cached = self.cache.get(key)
            if cached is not None:
                self.hits += 1
                if not is_leaf and key in self.depth_limited_cache:
                    self.horizon_visits += 1
                return cached

        if is_leaf:
            return self._remember(key, self.leaf_expectation(board))

        probability_4 = probability * 0.1 / count
        horizon_start = self.horizon_visits
        total = 0.0
        for position in positions:
            with_two = spawn(board, position, 2)
            with_four = spawn(board, position, 4)
            total += 0.9 * self.player(with_two, depth, probability_2)
            total += 0.1 * self.player(with_four, depth, probability_4)
        return self._remember(key, total / count, horizon_start)

    def _visit(self) -> None:
        self.nodes += 1
        if self.node_limit is not None and self.nodes > self.node_limit:
            raise SearchStopped("node limit")
        # Đọc đồng hồ mỗi 64 nút để giảm chi phí trong đệ quy.
        if self.nodes % 64 == 0:
            if self.deadline is not None and perf_counter() >= self.deadline:
                raise SearchStopped("deadline")

    def _remember(self, key, value, horizon_start=None):
        # Chỉ gọi sau khi tính xong nút; nút bị ngắt không được ghi vào cache.
        if self.cache_enabled and len(self.cache) < self.max_cache:
            self.cache[key] = value
            if horizon_start is not None and self.horizon_visits > horizon_start:
                self.depth_limited_cache.add(key)
        return value
