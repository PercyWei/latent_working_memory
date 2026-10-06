"""v3 仅使用项目或终端显式提供的凭据，避免共享账号的登录状态。"""

import os
from pathlib import Path

from dotenv import dotenv_values


def swanlab_api_key():
    """从项目根目录运行；有 .env 时只使用文件中的 key。"""
    path = Path(".env")
    if path.exists():
        key = dotenv_values(path, interpolate=False).get("SWANLAB_API_KEY")
        source = "project .env"
    else:
        key = os.environ.get("SWANLAB_API_KEY")
        source = "terminal environment"
    if key is None or not key.strip():
        raise ValueError(
            f"online tracking requires a nonempty SWANLAB_API_KEY in {source}; "
            "shared saved login credentials are not used"
        )
    return key.strip()
