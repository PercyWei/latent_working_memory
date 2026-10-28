"""父进程集中展示 teacher 准备进度，worker 仅共享少量整数。"""

import shutil
import sys
import time


_PHASES = ("loading", "prefill", "generating", "saving", "done")


class WorkerProgress:
    """每个 worker 独占写入；无锁读取允许瞬时误差，避免进程终止后遗留锁。"""

    def __init__(self, context):
        self.state = context.Array("q", 5, lock=False)

    def set_phase(self, phase, write_index=0):
        self.state[0] = _PHASES.index(phase)
        self.state[1] = write_index
        self.state[2] = 0

    def set_prompts(self, completed):
        self.state[2] = completed

    def finish_boundary(self):
        self.state[3] += 1

    def finish_document(self):
        self.state[4] += 1

    def snapshot(self):
        return tuple(self.state)


class PreparationProgress:
    def __init__(self, shards, workers, total_documents, total_boundaries, num_prompts, stream=None):
        self.shards = shards
        self.workers = workers
        self.total_documents = total_documents
        self.total_boundaries = total_boundaries
        self.num_prompts = num_prompts
        self.stream = sys.stderr if stream is None else stream
        self.tty = self.stream.isatty()
        self.boundaries = [sum(len(job.boundaries) for job in shard) for shard in shards]
        self.cached_documents = total_documents - sum(map(len, shards))
        self.cached_boundaries = total_boundaries - sum(self.boundaries)
        self.started = time.monotonic()
        self.last_show = None
        self.drawn_lines = 0

    def show(self, force=False, status="running"):
        now = time.monotonic()
        interval = 1 if self.tty else 30
        if not force and self.last_show is not None and now - self.last_show < interval:
            return
        self.last_show = now
        snapshots = [worker.snapshot() for worker in self.workers]
        documents = self.cached_documents + sum(state[4] for state in snapshots)
        boundaries = self.cached_boundaries + sum(state[3] for state in snapshots)
        percent = 100 * boundaries / self.total_boundaries if self.total_boundaries else 100
        seconds = int(now - self.started)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        elapsed = f"{hours:02}:{minutes:02}:{seconds:02}"
        lines = [
            f"teacher {status} | docs {documents}/{self.total_documents} | "
            f"boundaries {boundaries}/{self.total_boundaries} ({percent:.1f}%) | "
            f"cached {self.cached_boundaries} | elapsed {elapsed}"
        ]
        for rank, (shard, state) in enumerate(zip(self.shards, snapshots)):
            phase, write_index, prompts, saved, completed = state
            line = (
                f"cuda:{rank} | docs {completed}/{len(shard)} | "
                f"boundaries {saved}/{self.boundaries[rank]} | {_PHASES[phase]} "
                f"write={write_index}"
            )
            if _PHASES[phase] == "generating":
                line += f" prompts={prompts}/{self.num_prompts}"
            if completed < len(shard):
                line += f" doc={shard[completed].document_id[:8]}"
            lines.append(line)
        if self.tty:
            # 状态和文档哈希均为 ASCII；留一列，避免终端自动换行。
            width = max(1, shutil.get_terminal_size(fallback=(120, 24)).columns - 1)
            if self.drawn_lines:
                self.stream.write(f"\x1b[{self.drawn_lines}A")
            for line in lines:
                self.stream.write(f"\r\x1b[2K{line[:width]}\n")
            self.drawn_lines = len(lines)
        else:
            self.stream.write(" | ".join(lines) + "\n")
        self.stream.flush()
