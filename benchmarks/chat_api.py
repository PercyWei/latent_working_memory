"""Benchmark 共用的 Chat Completions 请求接口。"""

from functools import cache
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import dotenv_values
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential


@cache
def _project_credentials():
    """读取项目根目录的 .env，供进程内请求复用。"""
    values = dotenv_values(Path(__file__).resolve().parents[1] / ".env", interpolate=False)
    return values.get("OPENAI_API_KEY"), values.get("OPENAI_BASE_URL")


def _transient_error(error):
    if isinstance(error, HTTPError):
        return error.code in (408, 429) or 500 <= error.code < 600
    return isinstance(error, (URLError, TimeoutError, ConnectionError))


@retry(retry=retry_if_exception(_transient_error), wait=wait_random_exponential(min=1, max=30),
       stop=stop_after_attempt(5), reraise=True)
def chat_completion(model, messages, **options):
    """发送一条 Chat Completions 请求，返回解码后的 JSON 响应。"""
    file_key, file_base_url = _project_credentials()
    key = os.environ.get("OPENAI_API_KEY", file_key)
    if not key:
        raise ValueError("Set OPENAI_API_KEY in the environment or the project root .env")
    base = os.environ.get("OPENAI_BASE_URL", file_base_url) or "https://api.openai.com/v1"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if organization := os.environ.get("OPENAI_ORGANIZATION"):
        headers["OpenAI-Organization"] = organization
    body = {"model": model, "messages": messages, **options}
    url = f"{base.rstrip('/')}/chat/completions"
    request = Request(url, data=json.dumps(body).encode(), headers=headers)
    with urlopen(request, timeout=90) as response:
        return json.load(response)
