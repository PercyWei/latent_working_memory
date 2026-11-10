"""Local data adapters for the pinned E²-TTT evaluation harness."""

import json
from pathlib import Path

from datasets import Dataset
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks.ruler.common_utils import aggregate_metrics, process_results
from lm_eval.tasks.squad_completion.task import SQUADCompletion
from lm_eval.tasks.swde.task import SWDE


class LocalParquetMixin:
    """Replace dataset loading while inheriting the original task behavior."""

    def __init__(self, parquet_path, num_shards=1, shard_index=0):
        self.parquet_path = Path(parquet_path)
        self.num_shards, self.shard_index = num_shards, shard_index
        # The original class initializers discard custom config kwargs.
        ConfigurableTask.__init__(
            self,
            config={
                "task": self.task_name,
                "output_type": "generate_until",
                "validation_split": "validation",
                "doc_to_text": "text",
                "doc_to_target": "value",
                "num_fewshot": 0,
                "generation_kwargs": {
                    "do_sample": False,
                    "temperature": 0.0,
                    "until": ["\n"],
                    "max_gen_toks": 48,
                },
                "process_results": self.process_results.__func__,
                "metric_list": [
                    {
                        "metric": "contains",
                        "aggregation": self.aggregation()["contains"],
                        "higher_is_better": True,
                    }
                ],
                "metadata": {"version": self.VERSION},
            },
        )

    def download(self, dataset_kwargs=None, **kwargs):
        full = Dataset.from_parquet(str(self.parquet_path))
        if len(full) != self.source_samples:
            raise ValueError(f"{self.parquet_path}: expected {self.source_samples} samples, got {len(full)}")
        full = full.add_column("source_doc_id", list(range(len(full))))
        self.dataset = {"validation": full.select(range(self.shard_index, len(full), self.num_shards))}


class LocalSQuAD(LocalParquetMixin, SQUADCompletion):
    task_name = "squad_completion"
    source_samples = 2984


class LocalSWDE(LocalParquetMixin, SWDE):
    task_name = "swde"
    source_samples = 1111


class LocalRuler(ConfigurableTask):
    def __init__(self, data_path, task_name, length, lm, num_shards=1, shard_index=0):
        self.data_path = Path(data_path)
        self.length = length
        self.lm = lm
        self.metric = str(length)
        self.num_shards, self.shard_index = num_shards, shard_index
        super().__init__(
            config={
                "task": f"{task_name}_{length}",
                "output_type": "generate_until",
                "test_split": "test",
                "doc_to_text": "input",
                "doc_to_target": "outputs",
                "gen_prefix": None,
                "num_fewshot": 0,
                "generation_kwargs": {
                    "do_sample": False,
                    "temperature": 0.0,
                    "until": [],
                    "max_gen_toks": 128,
                },
                "process_results": process_results,
                "metric_list": [
                    {
                        "metric": self.metric,
                        "aggregation": aggregate_metrics,
                        "higher_is_better": True,
                    }
                ],
                "metadata": {"version": 1.0},
            }
        )

    def download(self, dataset_kwargs=None, **kwargs):
        with self.data_path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        if len(rows) != 500:
            raise ValueError(f"{self.data_path}: expected 500 RULER samples, got {len(rows)}")
        for index, row in enumerate(rows):
            row["max_length"] = self.length
            row["source_doc_id"] = index
        full = Dataset.from_list(rows)
        self.dataset = {"test": full.select(range(self.shard_index, len(full), self.num_shards))}

    def doc_to_text(self, doc):
        prompt = doc["input"] + doc["answer_prefix"]
        prompt_tokens = len(self.lm.tok_encode(prompt))
        if prompt_tokens + 128 > self.length:
            raise ValueError(
                f"{self.data_path}: input {prompt_tokens} + generation 128 exceeds "
                f"length {self.length}; RULER inputs must not be truncated."
            )
        return prompt

    def process_results(self, doc, results):
        scores = process_results(doc, results)
        return {self.metric: scores[self.metric]}
