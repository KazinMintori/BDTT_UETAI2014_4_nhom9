"""Hàm đánh giá Heuristic chuẩn mực cho game 2048.

Bao gồm 4 tiêu chí kinh điển, trực quan và dễ giải thích:
1. Số ô trống (Empty cells): Giữ không gian di chuyển thông thoáng.
2. Tính đơn điệu (Monotonicity): Khuyến khích các ô tăng dần về một góc.
3. Khả năng gộp (Merges): Khuyến khích các ô bằng nhau đứng cạnh nhau.
4. Ô lớn nhất ở góc (Max tile in corner): Giữ ô có giá trị lớn nhất ở góc.
"""

from .board import empty_count, is_terminal

# Trọng số của các tiêu chí đánh giá
WEIGHT_EMPTY = 270.0
WEIGHT_MONO = 45.0
WEIGHT_MERGES = 50.0
WEIGHT_CORNER = 100.0


def log2(val: int) -> int:
    """Trả về số mũ cơ số 2 của ô (ví dụ: 2->1, 4->2, 8->3, 1024->10)."""
    return val.bit_length() - 1 if val > 0 else 0


def monotonicity_score(board: tuple) -> float:
    """Tính độ đơn điệu theo hàng và theo cột.
    
    Phạt nếu các giá trị trên hàng hoặc cột không xếp theo thứ tự tăng/giảm dần.
    """
    penalty = 0.0

    # 1. Đơn điệu theo hàng
    for r in range(4):
        row = [log2(board[r * 4 + c]) for c in range(4)]
        inc = sum(max(0, row[i + 1] - row[i]) for i in range(3))
        dec = sum(max(0, row[i] - row[i + 1]) for i in range(3))
        penalty += min(inc, dec)

    # 2. Đơn điệu theo cột
    for c in range(4):
        col = [log2(board[r * 4 + c]) for r in range(4)]
        inc = sum(max(0, col[i + 1] - col[i]) for i in range(3))
        dec = sum(max(0, col[i] - col[i + 1]) for i in range(3))
        penalty += min(inc, dec)

    return -penalty


def merge_score(board: tuple) -> float:
    """Đếm số cặp ô kề nhau có cùng giá trị (khả năng gộp)."""
    merges = 0
    for r in range(4):
        for c in range(4):
            val = board[r * 4 + c]
            if val != 0:
                # Kề nhau theo hàng ngang
                if c < 3 and val == board[r * 4 + c + 1]:
                    merges += 1
                # Kề nhau theo hàng dọc
                if r < 3 and val == board[(r + 1) * 4 + c]:
                    merges += 1
    return float(merges)


def corner_score(board: tuple) -> float:
    """Thưởng điểm nếu ô lớn nhất nằm ở 1 trong 4 góc bàn cờ."""
    max_val = max(board)
    corners = (board[0], board[3], board[12], board[15])
    return float(log2(max_val)) if max_val in corners else 0.0


class Heuristic:
    def __init__(self, loss_penalty: float = 1e6):
        self.loss_penalty = loss_penalty

    def evaluate(self, board: tuple) -> float:
        """Đánh giá tổng thể trạng thái bàn cờ."""
        if is_terminal(board):
            return -self.loss_penalty

        empty = empty_count(board)
        mono = monotonicity_score(board)
        merges = merge_score(board)
        corner = corner_score(board)

        return (
            WEIGHT_EMPTY * empty
            + WEIGHT_MONO * mono
            + WEIGHT_MERGES * merges
            + WEIGHT_CORNER * corner
        )
