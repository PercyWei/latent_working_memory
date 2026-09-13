"""The output contract for model-independent PersonaMem construction."""

from collections import Counter
import json
from pathlib import Path
import re
import shutil
import statistics

from latent_working_memory.data_preparation.personamem.blocks import evidence_blocks
from latent_working_memory.data_preparation.personamem.common import now
from latent_working_memory.data_preparation.personamem.construction import save


DATA_FILES = frozenset(
    {
        "README.md",
        "metadata.json",
        "qas.jsonl",
        "qas.provisional.jsonl",
        "sources.json",
        "selection.json",
        "candidates.json",
        "source_exclusions.json",
    }
)
RUN_FILES = frozenset(
    {
        "completion.json",
        "config.json",
        "construction.lock",
        "content_rejections.json",
        "cross_split_removals.json",
        "dedup.log",
        "diagnostic.log",
        "diagnostic_panel.json",
        "diagnostic_results.json",
        "diagnostic_summary.json",
        "exact_duplicates.json",
        "final_diagnostic.log",
        "final_summary.json",
        "full.launch.json",
        "full.log",
        "full.summary.json",
        "pilot.launch.json",
        "pilot.log",
        "pilot.summary.json",
        "pilot_gate.json",
        "pilot_review.json",
        "pilot_review.log",
        "progress.json",
        "provenance.json",
        "requests.jsonl",
        "semantic_removals.json",
        "source_review.json",
        "source_review_panel.json",
    }
)
# These are actual operator/development records from the current run. Never fabricate them.
OPTIONAL_RECORDS = frozenset(
    {
        "manual_review.json",
        "tests.log",
        "tests_final_code.log",
        "tests_resume.log",
    }
)
RUN_DIRS = frozenset(
    {
        "requests",
        "results",
        "final_diagnostic",
        "source_code",
        "full_source_code",
        "final_source_code",
    }
)
DIAGNOSTIC_FILES = frozenset(
    {
        "diagnostic_panel.json",
        "diagnostic_results.json",
        "diagnostic_summary.json",
        "requests.jsonl",
    }
)
IGNORED_OS_FILES = frozenset({".DS_Store"})
CONFIG_KEYS = frozenset(
    {
        "endpoint",
        "model",
        "reasoning_effort",
        "max_output_tokens",
        "timeout_seconds",
        "concurrency",
        "max_attempts",
        "seed",
        "split_users",
        "pilot_candidates",
        "target_qas",
        "dataset_dir",
        "artifacts_dir",
        "source_csv",
        "history_cache",
        "source_url",
        "excluded_from_evaluation",
        "diagnostic_questions",
    }
)
REQUEST_FILE = re.compile(
    r"(?:candidate-\d{5,}|diagnostic-\d{3,}|pilot-review-\d{3,}|dedup-user-\d+)/"
    r"(?:generate|verify|gold|wrong|question_only|review|dedup)\."
    r"(?:request|parsed|attempt\d+\.(?:response|meta))\.json"
)


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_directory(root, files, directories, optional=(), complete=False):
    if not root.exists():
        if complete:
            raise ValueError(f"missing artifact directory: {root}")
        return
    present = {p.name for p in root.iterdir() if p.name not in IGNORED_OS_FILES}
    unexpected = present - set(files) - set(directories) - set(optional)
    missing = (set(files) | set(directories)) - present if complete else set()
    if unexpected or missing:
        raise ValueError(
            f"artifact contract mismatch at {root}: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for p in root.iterdir():
        if p.is_symlink():
            raise ValueError(f"artifact symlink not allowed: {p}")
        if p.name in directories and not p.is_dir():
            raise ValueError(f"expected artifact directory: {p}")
        if p.name in files and not p.is_file():
            raise ValueError(f"expected artifact file: {p}")


def check_layout(config, complete=False):
    if set(config) != CONFIG_KEYS:
        raise ValueError(
            f"construction config fields differ: missing={sorted(CONFIG_KEYS - set(config))}, "
            f"unexpected={sorted(set(config) - CONFIG_KEYS)}"
        )
    if not 1 <= config["concurrency"] <= 8 or config["diagnostic_questions"] < 1:
        raise ValueError("invalid construction concurrency or diagnostic question count")
    dataset, run = Path(config["dataset_dir"]), Path(config["artifacts_dir"])
    if dataset.resolve().is_relative_to(run.resolve()) or run.resolve().is_relative_to(
        dataset.resolve()
    ):
        raise ValueError("run records must be separate from the canonical dataset")
    check_directory(dataset, DATA_FILES, {"raw", "histories"}, complete=complete)
    check_directory(run, RUN_FILES, RUN_DIRS, OPTIONAL_RECORDS, complete)
    if (dataset / "sources.json").exists():
        users = {u["persona_id"] for u in read_json(dataset / "sources.json")["histories"]}
        if any(not u.isdigit() for u in users):
            raise ValueError("invalid PersonaMem source user ID")
        check_directory(dataset / "histories", {f"{u}.json" for u in users}, (), complete=complete)
        check_directory(
            dataset / "raw",
            {"persona_train.csv", *(f"{u}.json" for u in users)},
            (),
            complete=complete,
        )
    if (run / "results").exists():
        for p in (run / "results").iterdir():
            if p.name not in IGNORED_OS_FILES and not re.fullmatch(
                r"candidate-\d{5,}\.json", p.name
            ):
                raise ValueError(f"unexpected candidate artifact: {p}")
    for parent in [run, run / "final_diagnostic"]:
        if parent.name == "final_diagnostic":
            check_directory(parent, DIAGNOSTIC_FILES, {"requests"}, complete=complete)
        requests = parent / "requests"
        if requests.exists():
            for p in requests.rglob("*"):
                if p.is_file() and p.name not in IGNORED_OS_FILES:
                    if not REQUEST_FILE.fullmatch(p.relative_to(requests).as_posix()):
                        raise ValueError(f"unexpected model request artifact: {p}")


def snapshot_source(run, name):
    folder = run / name
    if folder.exists():
        return
    folder.mkdir(parents=True)
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copyfile(path, folder / path.name)


def verify_request_records(root):
    expected = set()
    for record in read_rows(root / "requests.jsonl"):
        folder = root / "requests" / record["key"]
        stage, attempt = record["stage"], record["attempt"]
        expected.update(
            (folder / f"{stage}.request.json", folder / f"{stage}.attempt{attempt}.meta.json")
        )
        response = folder / f"{stage}.attempt{attempt}.response.json"
        if record["ok"] or record.get("response_status") is not None or response.exists():
            expected.add(response)
        if record["ok"]:
            expected.add(folder / f"{stage}.parsed.json")
    actual = {
        p for p in (root / "requests").rglob("*") if p.is_file() and p.name not in IGNORED_OS_FILES
    }
    if expected != actual:
        raise ValueError(
            f"request artifact mismatch: missing={sorted(str(p) for p in expected - actual)}, "
            f"unexpected={sorted(str(p) for p in actual - expected)}"
        )


def summarize(config):
    dataset, run = Path(config["dataset_dir"]), Path(config["artifacts_dir"])
    sources = read_json(dataset / "sources.json")
    candidates = read_json(dataset / "candidates.json")
    results = [read_json(p) for p in sorted((run / "results").glob("*.json"))]
    if len(results) != len(candidates) or any(not r["ok"] for r in results):
        raise ValueError("cannot complete a dataset with unfinished or failed candidates")
    if {r["candidate_id"] for r in results} != {c["candidate_id"] for c in candidates}:
        raise ValueError("completed candidate identities do not match the selected sources")
    qas = read_rows(dataset / "qas.jsonl")
    final = read_json(run / "final_summary.json")
    if len(qas) != final["final_qas"] or len({q["qa_id"] for q in qas}) != len(qas) or not qas:
        raise ValueError("final QA identities/count disagree with the completion record")
    if dict(Counter(q["split"] for q in qas)) != final["by_split"]:
        raise ValueError("final QA split counts disagree")
    by_user = {}
    for q in qas:
        by_user.setdefault(q["persona_id"], []).append(q)
    for user in sources["histories"]:
        history = read_json(dataset / "histories" / f"{user['persona_id']}.json")
        rows = by_user.pop(user["persona_id"], [])
        if any(q["split"] != user["split"] or q["history_id"] != user["history_id"] for q in rows):
            raise ValueError("QA source or split mismatch")
        evidence_blocks(history, rows)
    if by_user:
        raise ValueError("QA contains an unknown source user")
    requests = read_rows(run / "requests.jsonl") + read_rows(
        run / "final_diagnostic/requests.jsonl"
    )
    core = read_json(run / "full.summary.json")
    pilot = read_json(run / "pilot.summary.json")
    counts = Counter(q["persona_id"] for q in qas)
    previous = (
        read_json(dataset / "metadata.json")
        if (dataset / "metadata.json").exists()
        else (read_json(run / "completion.json") if (run / "completion.json").exists() else {})
    )
    return dict(
        format="personamem-fact-qa-v1",
        completed_at=previous.get("completed_at", now()),
        source=config["source_url"].split("/resolve/")[0],
        history_variant="32k",
        users=len(sources["histories"]),
        users_by_split=dict(Counter(h["split"] for h in sources["histories"])),
        qas=len(qas),
        qas_by_split=final["by_split"],
        qa_per_user=dict(
            min=min(counts.values()),
            median=statistics.median(counts.values()),
            max=max(counts.values()),
        ),
        subject_types=final["by_subject_type"],
        fact_types=final["by_fact_type"],
        concurrency=config["concurrency"],
        candidates=len(candidates),
        generated_qas=core["generated_qas"],
        program_rejections=core["program_rejections"],
        semantic_verification_rejections=core["semantic_rejections"],
        pilot_content_rejections=core["content_rejections"],
        user_audit_rejections=final["semantic_removals"],
        exact_duplicates=core["duplicates"],
        cross_split_duplicates=final["cross_split_removals"],
        annotation_requests=sum(r["stage"] in {"generate", "verify"} for r in requests),
        total_model_requests=len(requests),
        failed_requests=sum(not r["ok"] for r in requests),
        request_counts_by_stage=dict(Counter(r["stage"] for r in requests)),
        generation_and_verification_wall_seconds=pilot["elapsed_this_run"]
        + core["elapsed_this_run"],
        test_diagnostic=read_json(run / "final_diagnostic/diagnostic_summary.json"),
        files=dict(
            qas="qas.jsonl",
            histories="histories/",
            sources="sources.json",
            selection="selection.json",
        ),
        annotation_model=config["model"],
        annotation_reasoning_effort=config["reasoning_effort"],
        preprocessing="Original text and character evidence are model-independent; each experiment "
        "supplies its own tokenizer and computes token positions in memory.",
        limitations=[
            "Model-based annotation checks are not independent human or training-model accuracy.",
            "Historically anchored QA do not prove that the full histories are conflict-free.",
        ],
    )


def render_readme(metadata):
    lines = [
        "# PersonaMem-v2 事实 QA：模型无关数据",
        "",
        "本目录保存所有模型共用的原始对话、QA、消息与字符证据和用户划分。",
        "",
        "| 划分 | 完整历史 | 最终 QA |",
        "|---|---:|---:|",
    ]
    for split, users in metadata["users_by_split"].items():
        lines.append(f"| {split} | {users} | {metadata['qas_by_split'].get(split, 0)} |")
    lines += [
        f"| 合计 | {metadata['users']} | {metadata['qas']} |",
        "",
        "- `qas.jsonl`：最终 QA；`histories/`：原始 user/assistant 消息。",
        "- `sources.json`、`selection.json`：来源、划分及用户排除名单。",
        "- `metadata.json`：规模、标注模型来源和质量统计。",
        "- `candidates.json`、`source_exclusions.json`、`raw/`：候选与原始来源记录。",
        "- `qas.provisional.jsonl`：审查前中间记录，不作为最终训练 QA。",
        "",
        "答案由消息 ID 和 Unicode 字符区间定位，区间右端不包含；完整证据保存在 `evidence_messages`。",
        "写作材料人物、第三方和用户本人分别标记，不能把材料内容直接当成用户真实经历。",
        "",
        "构造入口：`python -m latent_working_memory.data_preparation.personamem --config <配置>`。",
        "离线检查已有交付：同一命令增加 `--stage verify`。",
        "运行时读取：`PersonaMemDataset(dataset_dir, tokenizer)`，不会在数据目录写入分词结果。",
        "",
        "本次交付不生成测速、模型专属派生数据、训练配置或共享评估计划。",
        "运行记录目录保存请求、响应、逐片段结果、审查和诊断、阶段日志及源码快照。",
        "人工复核和开发测试日志只在实际执行后保存，不由构造程序伪造。",
        "",
        "固定 test 样本的 API 可回答性检查：",
        "",
        "| 条件 | EM | F1 |",
        "|---|---:|---:|",
    ]
    for name, scores in metadata["test_diagnostic"]["conditions"].items():
        lines.append(f"| {name} | {scores['em']:.2%} | {scores['f1']:.2%} |")
    lines += ["", "该检查来自标注模型，不是后续训练模型或独立人工的准确率。", ""]
    return "\n".join(lines)


def finalize(config):
    check_layout(config)
    metadata = summarize(config)
    dataset, run = Path(config["dataset_dir"]), Path(config["artifacts_dir"])
    required = [
        *(dataset / name for name in DATA_FILES - {"README.md", "metadata.json"}),
        *(run / name for name in RUN_FILES - {"completion.json"}),
        *(run / name for name in RUN_DIRS),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"cannot publish incomplete construction artifacts: {missing}")
    check_directory(run / "final_diagnostic", DIAGNOSTIC_FILES, {"requests"}, complete=True)
    for root in (run, run / "final_diagnostic"):
        verify_request_records(root)
    # Do not overwrite reviewed documentation or historical metadata on a verified rerun.
    if not (dataset / "README.md").exists():
        (dataset / "README.md").write_text(render_readme(metadata))
    if not (dataset / "metadata.json").exists():
        save(dataset / "metadata.json", metadata)
    if not (run / "completion.json").exists():
        save(run / "completion.json", metadata)
    verify(config)
    return metadata


def verify(config):
    check_layout(config, complete=True)
    if read_json(Path(config["artifacts_dir"]) / "config.json") != config:
        raise ValueError("construction configuration differs from its saved run")
    for root in (Path(config["artifacts_dir"]), Path(config["artifacts_dir"]) / "final_diagnostic"):
        verify_request_records(root)
    metadata = summarize(config)
    for path in [
        Path(config["dataset_dir"]) / "metadata.json",
        Path(config["artifacts_dir"]) / "completion.json",
    ]:
        actual = read_json(path)
        for key in [
            "users",
            "users_by_split",
            "qas",
            "qas_by_split",
            "candidates",
            "generated_qas",
            "total_model_requests",
            "annotation_model",
        ]:
            if actual[key] != metadata[key]:
                raise ValueError(f"artifact metadata mismatch: {path}: {key}")
    return dict(users=metadata["users"], qas=metadata["qas"], artifact_contract="passed")
