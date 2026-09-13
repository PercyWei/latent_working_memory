"""Prepare fixed, user-isolated PersonaMem-v2 sources for factual annotation."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import random
import shutil
import time
from urllib.parse import quote
from urllib.request import urlopen

from latent_working_memory.data_preparation.personamem.common import now, write_json


def prepare(config):
    root = Path(config["dataset_dir"])
    root.mkdir(parents=True, exist_ok=True)
    prepared = root / "sources.json"
    if prepared.exists():
        result = json.loads(prepared.read_text())
        if result["seed"] != config["seed"] or result["split_user_counts"] != config["split_users"]:
            raise ValueError("existing sources disagree with configuration")
        return result
    with Path(config["source_csv"]).open() as handle:
        rows = list(csv.DictReader(handle))
    by_user = {}
    for index, row in enumerate(rows):
        by_user.setdefault(row["persona_id"], []).append((index, row))
    excluded_users = set(config["excluded_from_evaluation"])
    rng = random.Random(config["seed"])
    users = sorted(by_user, key=int)
    rng.shuffle(users)
    assignments = {
        user: "train"
        if i < int(0.8 * len(users))
        else "dev"
        if i < int(0.9 * len(users))
        else "test"
        for i, user in enumerate(users)
    }
    chosen = []
    for split, count in config["split_users"].items():
        pool = [
            u
            for u in users
            if assignments[u] == split and (split == "train" or u not in excluded_users)
        ]
        rng.shuffle(pool)
        if len(pool) < count:
            raise ValueError("insufficient users for held-out selection")
        chosen.extend(dict(persona_id=u, split=split) for u in pool[:count])
    selection_path = root / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection["seed"] != config["seed"] or selection["split_users"] != config["split_users"]:
            raise ValueError("selection/config mismatch")
        if (
            selection["users"] != chosen
            or set(selection["excluded_from_evaluation"]) != excluded_users
        ):
            raise ValueError("saved user selection differs from source/configuration")
    else:
        write_json(
            selection_path,
            dict(
                seed=config["seed"],
                split_users=config["split_users"],
                users=chosen,
                all_train_source_user_splits=assignments,
                excluded_from_evaluation=sorted(excluded_users, key=int),
                created_at=now(),
            ),
        )
    history_dir = root / "histories"
    raw_dir = root / "raw"
    history_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    csv_target = raw_dir / "persona_train.csv"
    if not csv_target.exists():
        shutil.copyfile(config["source_csv"], csv_target)

    def load_one(user):
        uid = user["persona_id"]
        paths = {r["chat_history_32k_link"] for _, r in by_user[uid]}
        if len(paths) != 1:
            raise ValueError(f"ambiguous training history for {uid}")
        source_path = paths.pop()
        local_raw = raw_dir / f"{uid}.json"
        if not local_raw.exists():
            cache = Path(config["history_cache"]) / f"{uid}.json"
            if cache.exists():
                raw = json.loads(cache.read_text())
            else:
                url = f"{config['source_url']}/{quote(source_path, safe='/')}"
                for attempt in range(3):
                    try:
                        with urlopen(url, timeout=60) as response:
                            raw = json.load(response)
                        break
                    except OSError:
                        if attempt == 2:
                            raise
                        time.sleep(2**attempt)
            write_json(local_raw, raw)
        else:
            raw = json.loads(local_raw.read_text())
        if str(raw["metadata"]["persona_id"]) != uid:
            raise ValueError(f"history persona mismatch: {uid}")
        messages = [
            dict(message_id=f"m{i:04d}", role=m["role"], content=m["content"])
            for i, m in enumerate(raw["chat_history"])
            if m["role"] in {"user", "assistant"}
        ]
        if not messages or any(not isinstance(m["content"], str) for m in messages):
            raise ValueError("invalid history messages")
        history_id = f"personamem-v2:32k:{uid}"
        record = dict(
            user,
            history_id=history_id,
            source_history=source_path,
            messages=messages,
            raw_file=f"raw/{uid}.json",
        )
        write_json(history_dir / f"{uid}.json", record)
        pairs = [(m["role"], m["content"]) for m in messages]
        candidates = {}
        exclusions = []
        for index, row in by_user[uid]:
            snippet = json.loads(row["related_conversation_snippet"])
            target = [(m["role"], m["content"]) for m in snippet]
            matches = (
                [
                    i
                    for i in range(len(pairs) - len(target) + 1)
                    if pairs[i : i + len(target)] == target
                ]
                if target
                else []
            )
            if row["pref_type"] in {"ask_to_forget", "sensitive_info"}:
                exclusions.append(dict(source_row=index, reason=row["pref_type"]))
                continue
            if len(matches) != 1:
                exclusions.append(dict(source_row=index, reason=f"snippet_matches={len(matches)}"))
                continue
            start, end = matches[0], matches[0] + len(target)
            key = f"{history_id}:{start}:{end}"
            if key in candidates:
                candidates[key]["source_rows"].append(index)
                continue
            candidates[key] = dict(
                user,
                history_id=history_id,
                candidate_id=key,
                source_rows=[index],
                source_history=source_path,
                message_start=start,
                message_end_exclusive=end,
                position_fraction=start / len(messages),
                scenario=row["conversation_scenario"],
                pref_type=row["pref_type"],
                messages=messages[start:end],
            )
        return record, list(candidates.values()), exclusions

    loaded = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(load_one, user): user for user in chosen}
        for future in as_completed(futures):
            record, candidates, exclusions = future.result()
            loaded[record["persona_id"]] = (record, candidates, exclusions)
            print(
                json.dumps(
                    dict(
                        source_user=record["persona_id"],
                        candidates=len(candidates),
                        completed_users=len(loaded),
                    )
                ),
                flush=True,
            )
    ordered, all_exclusions, histories = [], [], []
    for split in config["split_users"]:
        queues = []
        for user in chosen:
            if user["split"] != split:
                continue
            record, candidates, exclusions = loaded[user["persona_id"]]
            histories.append({k: v for k, v in record.items() if k != "messages"})
            all_exclusions.extend(dict(persona_id=user["persona_id"], **e) for e in exclusions)
            rng.shuffle(candidates)
            buckets = [
                [c for c in candidates if min(2, int(c["position_fraction"] * 3)) == b]
                for b in range(3)
            ]
            queue = []
            while any(buckets):
                for bucket in buckets:
                    if bucket:
                        queue.append(bucket.pop())
            queues.append(queue)
        while any(queues):
            for queue in queues:
                if queue:
                    ordered.append(queue.pop(0))
    write_json(root / "candidates.json", ordered)
    write_json(root / "source_exclusions.json", all_exclusions)
    result = dict(
        created_at=now(),
        seed=config["seed"],
        split_user_counts=config["split_users"],
        histories=histories,
        candidates=len(ordered),
        candidates_by_split={
            s: sum(c["split"] == s for c in ordered) for s in config["split_users"]
        },
        source_message_indices="original JSON indices, zero based",
        message_ranges="filtered user/assistant indices, right endpoint excluded",
    )
    write_json(prepared, result)
    return result
