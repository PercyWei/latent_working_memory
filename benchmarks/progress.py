"""终端进度使用本次运行新增的题数计算平均速度。"""

import time


class QuestionTimer:
    def __init__(self, initial_count=0):
        self.initial_count = initial_count
        self.started = time.monotonic()

    def format_rate(self, completed):
        elapsed = time.monotonic() - self.started
        per_minute = 60 * (completed - self.initial_count) / elapsed if elapsed > 0 else 0.0
        return f"{per_minute:.2f} q/min"
