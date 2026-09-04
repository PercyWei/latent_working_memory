from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdic_repro.config import RetrievalConfig
from cdic_repro.engine import CdicInferenceEngine
from cdic_repro.icae_adapter import (
    IcaeV1AdapterConfig,
    IcaeV1InferenceAdapter,
    torch_cosine_similarity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single dialogue through C-DIC")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--decay", type=float, default=0.05)
    parser.add_argument("--max-retrieved", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    adapter = IcaeV1InferenceAdapter.load(
        IcaeV1AdapterConfig(
            model_path=arguments.model_path,
            checkpoint_path=arguments.checkpoint,
            max_new_tokens=arguments.max_new_tokens,
            seed=arguments.seed,
        )
    )
    engine = CdicInferenceEngine(
        model=adapter,
        similarity=torch_cosine_similarity,
        retrieval_config=RetrievalConfig(
            threshold=arguments.threshold,
            decay=arguments.decay,
            max_retrieved=arguments.max_retrieved,
        ),
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        arguments.input.open(encoding="utf-8") as source,
        arguments.output.open("w", encoding="utf-8") as destination,
    ):
        for index, line in enumerate(source):
            if arguments.max_turns is not None and index >= arguments.max_turns:
                break
            record = json.loads(line)
            query = record["query"]
            query_id = str(record.get("id", f"turn-{index + 1:06d}"))
            output = engine.step(query, query_id=query_id)
            destination.write(
                json.dumps(
                    {
                        "id": query_id,
                        "query": query,
                        "response": output.response,
                        "trace": output.trace.to_dict(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
