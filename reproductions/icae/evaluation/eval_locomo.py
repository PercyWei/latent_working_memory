"""ICAE 的公共 LoCoMo 评估入口。"""

import sys
from latent_working_memory.benchmarks.locomo.evaluate import main

if __name__ == "__main__":
    main(["--method", "icae", "--output-root", "artifacts/reproductions/icae", *sys.argv[1:]])
