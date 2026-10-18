"""训练状态的跨进程读取与按字段更新。"""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path


def _read_status(root):
    path = root / "status.json"
    return (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file() else {"checkpoints": {}, "summaries": []}
    )


def read_training_status(root):
    """读取完整快照；尚未建立状态时不推断执行状态或阶段登记。"""
    root = Path(root)
    with (root / ".training-status.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        return _read_status(root)


@contextmanager
def update_training_status(root):
    """在同一进程锁内修改最新字段，成功退出后原子提交。"""
    root = Path(root)
    with (root / ".training-status.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status = _read_status(root)
        yield status
        temporary = root / "status.json.tmp"
        temporary.write_text(
            json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(root / "status.json")
