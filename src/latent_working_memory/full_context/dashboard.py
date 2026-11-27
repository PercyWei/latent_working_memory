"""按任务比较原始模型的全部已评分运行。"""

from benchmarks.checkpoint_dashboard import run_dashboard


if __name__ == "__main__":
    run_dashboard("full-context", "Full-context")
