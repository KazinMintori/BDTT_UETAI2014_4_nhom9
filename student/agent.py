"""Chọn nước đi bằng Expectimax và chia thời gian suy nghĩ cho cả ván."""

from time import perf_counter

from game2048.observation import Observation
from game2048.types import Move

from .board import chance_outcomes, empty_count, from_matrix, move
from .heuristic import Heuristic
from .search import Expectimax, SearchStopped

DEFAULT_CUTOFF = 0.006


class Agent:
    name = "expectimax-tuple-v6"

    def __init__(
        self,
        variant="space",
        max_depth=8,
        cache_enabled=True,
        budget_scale=1.0,
        reward_weight=0.1,
        fast_leaf=True,
        cutoff=DEFAULT_CUTOFF,
        loss_penalty=1e6,
        node_rate=160.0,
        unlimited_node_rate=320.0,
    ):
        self.variant = variant
        self.max_depth = max_depth
        self.budget_scale = budget_scale
        self.node_rate = node_rate
        self.unlimited_node_rate = unlimited_node_rate
        self.heuristic = Heuristic(variant, loss_penalty)
        self.search = Expectimax(
            self.heuristic,
            cutoff=cutoff,
            reward_weight=reward_weight,
            cache_enabled=cache_enabled,
        )
        if not fast_leaf:
            self.search.leaf_expectation = self._full_spawn_expectation
        self.reset()

    def reset(self, seed=None) -> None:
        # Agent không lấy mẫu ngẫu nhiên nên không cần dùng seed.
        self.search.reset()
        self._reset_move_stats()
        self.game_nodes = 0
        self.game_cache_hits = self.game_cache_lookups = 0
        self.game_search_ms = 0.0
        self.game_depth_counts = {}
        self.game_stop_counts = {}
        self.cache_peak_entries = 0

    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        started = perf_counter()
        if not obs.legal_moves:
            raise ValueError("choose_move called on terminal board")

        board = from_matrix(obs.board)
        legal_moves = tuple(sorted(obs.legal_moves))
        best_move, close_scores = self._fallback_move(board, legal_moves)
        self._reset_move_stats()
        self.search.reset()
        has_total_budget = obs.time_remaining_ms is not None

        budget_ms = self._time_budget_ms(obs, time_limit_ms, board, close_scores)
        self.last_slice_ms = budget_ms
        if len(legal_moves) == 1 or budget_ms < 0.25:
            self._record_move_stats()
            return best_move

        # Chừa thời gian cho việc đóng gói kết quả và sai số khi kiểm tra đồng hồ.
        usable_ms = max(0.0, budget_ms * 0.92 - 0.25)
        deadline = started + usable_ms / 1000.0
        # Không tăng tốc độ tiêu ngân sách hữu hạn chỉ vì chế độ UI còn dư
        # thời gian mỗi nước. Hai chế độ có ràng buộc tài nguyên khác nhau.
        rate = self.node_rate if has_total_budget else self.unlimited_node_rate
        raw_node_limit = max(64, int(usable_ms * rate * 0.9))
        node_limit = 64
        while node_limit * 2 <= raw_node_limit:
            node_limit *= 2
        self.search.configure_limits(deadline, node_limit)

        search_started = perf_counter()
        self.last_search_stop = "max depth"
        for depth in range(1, self.max_depth + 1):
            try:
                candidate, _ = self.search.root(board, depth, legal_moves)
            except SearchStopped as stopped:
                self.last_search_stop = str(stopped)
                break
            # Chỉ thay nước cũ khi đã xét xong mọi hướng ở độ sâu này.
            best_move = candidate
            self.last_completed_depth = depth
            if self.search.probability_saturated:
                self.last_search_stop = "probability cutoff"
                break

        self.last_search_ms = (perf_counter() - search_started) * 1000.0
        self.last_nodes = self.search.nodes
        self.last_cache_hits = self.search.hits
        self.last_cache_lookups = self.search.lookups
        self._record_move_stats()
        return best_move

    def _fallback_move(self, board: tuple, legal_moves) -> tuple[Move, bool]:
        """Chuẩn bị một nước hợp lệ để dùng nếu không còn thời gian tìm sâu."""
        values = []
        best_move = legal_moves[0]
        best_value = float("-inf")
        for direction in legal_moves:
            next_board, reward, changed = move(board, int(direction))
            if not changed:
                raise ValueError("observation legal moves disagree with board")
            value = self.heuristic.value(next_board) + self.search.reward_weight * reward
            values.append(value)
            if value > best_value:
                best_value = value
                best_move = direction

        values.sort(reverse=True)
        close_scores = len(values) > 1 and values[0] - values[1] < 1000.0
        return best_move, close_scores

    def _time_budget_ms(self, obs, time_limit_ms, board, close_scores) -> float:
        available_ms = max(0.0, float(time_limit_ms))
        if obs.time_remaining_ms is None:
            return min(available_ms * 0.8, 40.0)

        available_ms = min(available_ms, max(0.0, obs.time_remaining_ms))
        if available_ms <= 3.0:
            return 0.0

        # Hệ số dự trữ để tránh tiêu hết giờ ở đầu ván, không dự đoán số nước còn lại.
        reserve_moves = max(900.0, 3000.0 - 0.65 * obs.move_count)
        empty_cells = empty_count(board)
        if empty_cells >= 8:
            difficulty = 0.35
        elif empty_cells >= 5:
            difficulty = 0.65
        elif empty_cells >= 3:
            difficulty = 1.2
        else:
            difficulty = 2.0
        if close_scores:
            difficulty *= 1.2

        requested_ms = available_ms / reserve_moves * difficulty * self.budget_scale
        return min(requested_ms, 90.0, max(0.0, available_ms - 2.0))

    def _full_spawn_expectation(self, board: tuple) -> float:
        total = 0.0
        for next_board, probability in chance_outcomes(board):
            total += probability * self.heuristic.value(next_board)
        return total

    def _reset_move_stats(self) -> None:
        self.last_nodes = self.last_completed_depth = 0
        self.last_cache_hits = self.last_cache_lookups = 0
        self.last_search_ms = self.last_slice_ms = 0.0
        self.last_search_stop = "fallback"

    def _record_move_stats(self) -> None:
        self.game_nodes += self.last_nodes
        self.game_cache_hits += self.last_cache_hits
        self.game_cache_lookups += self.last_cache_lookups
        self.game_search_ms += self.last_search_ms
        depth = self.last_completed_depth
        self.game_depth_counts[depth] = self.game_depth_counts.get(depth, 0) + 1
        reason = self.last_search_stop
        self.game_stop_counts[reason] = self.game_stop_counts.get(reason, 0) + 1
        self.cache_peak_entries = max(self.cache_peak_entries, len(self.search.cache))
