"""运行与评估任务的占用锁；锁位于 run 删除范围之外。"""

from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def _lock(path, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, mode | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"an experiment is already running: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _lock_directory(run_directory):
    directory = Path(run_directory).expanduser().resolve()
    return directory.parent / ".locks" / directory.name


def run_lock(run_directory, exclusive=False):
    return _lock(_lock_directory(run_directory) / "run.lock",
                 fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)


def evaluation_lock(run_directory, eval_id, task):
    return _lock(_lock_directory(run_directory) / f"eval-{eval_id}" / f"{task}.lock", fcntl.LOCK_EX)
