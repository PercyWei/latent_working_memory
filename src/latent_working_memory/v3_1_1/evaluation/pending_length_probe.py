"""对冻结的边界输入训练长度探针，按文档调参并在原 dev 上报告结果。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import torch
from torch import nn

from ..io import write_json


def _now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d %H:%M:%S Asia/Shanghai")


def _load_documents(feature_dir):
    paths = sorted(Path(feature_dir).glob("shard_*.pt"))
    if not paths:
        raise ValueError("feature directory contains no shard_*.pt files")
    documents, metadata, indices = [], None, []
    seen = set()
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=True)
        current = shard["metadata"]
        common = {key: current[key] for key in (
            "checkpoint", "checkpoint_step", "seed", "temperature", "recent_size",
            "num_shards", "data_identity", "feature_dtype",
        )}
        if metadata is None:
            metadata = common
        elif metadata != common:
            raise ValueError("feature shards have different collection identities")
        indices.append(current["shard_index"])
        for document in shard["documents"]:
            identity = document["id"]
            if identity in seen:
                raise ValueError(f"document {identity} occurs more than once across splits/shards")
            seen.add(identity)
            features = document["features"]
            count = len(document["pending_tokens_before"])
            if (document["split"] not in {"train", "dev"}
                or features.ndim != 2 or len(features) != count
                or any(document[key].shape != (count,) for key in (
                    "after_first_compress", "read_tokens", "compress_actions",
                ))):
                raise ValueError(f"invalid feature/target alignment for {identity}")
            documents.append(document)
    if sorted(indices) != list(range(metadata["num_shards"])):
        raise ValueError("feature shards are missing or duplicated")
    documents.sort(key=lambda document: document["id"])
    train = [document for document in documents if document["split"] == "train"]
    dev = [document for document in documents if document["split"] == "dev"]
    if any(len(selected) != metadata["data_identity"][split]["documents"]
           for split, selected in (("train", train), ("dev", dev))):
        raise ValueError("feature document counts differ from the bound train/dev data identity")
    if len(train) < 5 or not dev:
        raise ValueError("probe requires at least five training documents and a held-out dev set")
    return train, dev, metadata


def _stack(documents, device):
    return {
        "document_ids": [document["id"] for document in documents],
        "document_index": torch.cat([
            torch.full((len(document["features"]),), index, dtype=torch.int64)
            for index, document in enumerate(documents)
        ]).to(device),
        "features": torch.cat([document["features"] for document in documents]).to(
            device=device, dtype=torch.float32,
        ),
        **{key: torch.cat([document[key] for document in documents]).to(device) for key in (
            "pending_tokens_before", "after_first_compress", "read_tokens", "compress_actions",
        )},
    }


def _scaler(data):
    features, target = data["features"], data["pending_tokens_before"].float()
    return {
        "feature_mean": features.mean(0),
        "feature_std": features.std(0, correction=0).clamp_min(1e-6),
        "target_mean": target.mean(),
        "target_std": target.std(correction=0).clamp_min(1e-6),
    }


def _normalize(data, scaler):
    return (
        (data["features"] - scaler["feature_mean"]) / scaler["feature_std"],
        (data["pending_tokens_before"].float() - scaler["target_mean"]) / scaler["target_std"],
    )


def _ridge_gram(features, targets):
    features, targets = features.double(), targets.double()
    return features.T @ features, features.T @ targets


def _ridge_solve(gram, rhs, alpha):
    regularized = gram.clone()
    regularized.diagonal().add_(alpha)
    return torch.linalg.solve(regularized, rhs)


def _read_position_fit(data):
    positions = data["read_tokens"].double()
    targets = data["pending_tokens_before"].double()
    centered = positions - positions.mean()
    slope = (centered * (targets - targets.mean())).sum() / centered.square().sum()
    return {"slope": slope, "intercept": targets.mean() - slope * positions.mean()}


def _new_mlp(width, device, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 1)).to(device)


def _fit_mlp(features, target, epochs, seed, log, stage, validation=None, patience=10):
    model = _new_mlp(features.shape[1], features.device, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_epoch, best_loss = 0, float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(features), generator=generator).to(features.device)
        total = 0.0
        for indices in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(features[indices]).squeeze(-1) - target[indices]).square().mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        row = {"stage": stage, "epoch": epoch, "train_batch_mse": total / len(features)}
        if validation is not None:
            model.eval()
            with torch.no_grad():
                value = (model(validation[0]).squeeze(-1) - validation[1]).square().mean().item()
            row["tune_mse"] = value
            if value < best_loss:
                best_loss, best_epoch = value, epoch
            row["best_epoch"] = best_epoch
        log.write(json.dumps(row, allow_nan=False) + "\n")
        log.flush()
        print(json.dumps(row, allow_nan=False), flush=True)
        if validation is not None and epoch - best_epoch >= patience:
            break
    model.eval()
    return model, best_epoch, best_loss


def _metrics(target, prediction):
    if not len(target):
        return {"r2": None, "mae_tokens": None}
    target, prediction = target.double(), prediction.double()
    residual = prediction - target
    total = (target - target.mean()).square().sum()
    return {
        "r2": 1 - residual.square().sum().item() / total.item() if total > 0 else None,
        "mae_tokens": residual.abs().mean().item(),
    }


def _summarize(data, predictions):
    target = data["pending_tokens_before"]
    scopes = {
        "overall": torch.ones_like(target, dtype=torch.bool),
        "before_first_compress": ~data["after_first_compress"],
        "after_first_compress": data["after_first_compress"],
    }
    result = {}
    for name, mask in scopes.items():
        selected = target[mask].float()
        result[name] = {
            "documents": len(data["document_index"][mask].unique()),
            "decisions": int(mask.sum()),
            "target_tokens": ({
                "min": selected.min().item(), "max": selected.max().item(),
                "mean": selected.mean().item(), "std": selected.std(correction=0).item(),
            } if len(selected) else None),
            "models": {key: _metrics(selected, prediction[mask])
                       for key, prediction in predictions.items()},
        }
    result["length_buckets"] = []
    for lower, upper in ((0, 1024), (1024, 2048), (2048, 4096), (4096, None)):
        mask = target >= lower
        if upper is not None:
            mask &= target < upper
        result["length_buckets"].append({
            "lower_inclusive": lower, "upper_exclusive": upper,
            "documents": len(data["document_index"][mask].unique()),
            "decisions": int(mask.sum()),
            "models": {key: _metrics(target[mask], prediction[mask])
                       for key, prediction in predictions.items()},
        })
    return result


def fit(feature_dir, output_dir, device="cpu", seed=42):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "plan.json").exists():
        raise FileExistsError("probe output already has a plan; use a fresh output directory")
    train_documents, dev_documents, metadata = _load_documents(feature_dir)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_documents), generator=generator).tolist()
    tune_count = max(1, round(len(train_documents) * 0.2))
    tune_documents = [train_documents[index] for index in order[:tune_count]]
    fit_documents = [train_documents[index] for index in order[tune_count:]]
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    plan = {
        "created_at": _now(), "feature_dir": str(Path(feature_dir).resolve()),
        "feature_metadata": metadata, "device": device, "seed": seed,
        "split": {"fit": [row["id"] for row in fit_documents],
                  "tune": [row["id"] for row in tune_documents],
                  "dev": [row["id"] for row in dev_documents]},
        "weighting": "Each autonomous decision has equal weight.",
        "normalization": "Feature z-scores and target z-scores fit on the fitting documents only; "
                         "refit on all train documents for the final models.",
        "ridge": {"alphas": alphas, "objective": "sum_squared_error + alpha * squared_weight_norm",
                  "selection": "minimum tune MSE; refit on all train documents", "dtype": "float64"},
        "mlp": {"hidden_size": 128, "activation": "SiLU", "optimizer": "AdamW", "lr": 1e-3,
                "weight_decay": 1e-4, "batch_size": 256, "max_epochs": 100, "patience": 10,
                "selection": "minimum tune MSE epoch; reinitialize with same seed and refit all train "
                             "for that number of epochs", "dtype": "float32"},
        "baselines": ["all-train mean pending length", "OLS(read_tokens), including intercept; "
                      "fit on all train decisions"],
        "stage_definition": "The first compress decision itself is before_first_compress; "
                            "subsequent autonomous decisions are after_first_compress.",
        "qa_scoring": False,
    }
    write_json(output_dir / "plan.json", plan)
    started = time.monotonic()
    fit_data, tune_data = _stack(fit_documents, device), _stack(tune_documents, device)
    inner_scaler = _scaler(fit_data)
    fit_x, fit_y = _normalize(fit_data, inner_scaler)
    tune_x, tune_y = _normalize(tune_data, inner_scaler)
    gram, rhs = _ridge_gram(fit_x, fit_y)
    alpha_scores = []
    for alpha in alphas:
        weights = _ridge_solve(gram, rhs, alpha)
        mse = ((tune_x.double() @ weights) - tune_y.double()).square().mean().item()
        alpha_scores.append({"alpha": alpha, "tune_mse": mse})
        print(json.dumps({"stage": "ridge_selection", **alpha_scores[-1]}), flush=True)
    chosen_alpha = min(alpha_scores, key=lambda row: row["tune_mse"])["alpha"]
    del gram, rhs, weights
    with (output_dir / "training.jsonl").open("w", encoding="utf-8") as log:
        inner_model, best_epoch, best_loss = _fit_mlp(
            fit_x, fit_y, 100, seed, log, "mlp_selection", validation=(tune_x, tune_y),
        )
        del inner_model, fit_data, tune_data, fit_x, fit_y, tune_x, tune_y, inner_scaler
        train_data = _stack(train_documents, device)
        dev_data = _stack(dev_documents, device)
        scaler = _scaler(train_data)
        train_x, train_y = _normalize(train_data, scaler)
        dev_x, _ = _normalize(dev_data, scaler)
        gram, rhs = _ridge_gram(train_x, train_y)
        ridge_weights = _ridge_solve(gram, rhs, chosen_alpha)
        del gram, rhs
        mlp, _, _ = _fit_mlp(train_x, train_y, best_epoch, seed, log, "mlp_refit")
    summaries, stored_predictions = {}, {}
    read_position = _read_position_fit(train_data)
    with torch.no_grad():
        for split, data, features in (("train", train_data, train_x), ("dev", dev_data, dev_x)):
            predictions = {
                "train_mean": torch.full_like(
                    data["pending_tokens_before"], scaler["target_mean"].item(), dtype=torch.float32,
                ),
                "read_position": data["read_tokens"].double() * read_position["slope"]
                                 + read_position["intercept"],
                "ridge": (features.double() @ ridge_weights) * scaler["target_std"]
                         + scaler["target_mean"],
                "mlp": mlp(features).squeeze(-1) * scaler["target_std"] + scaler["target_mean"],
            }
            summaries[split] = _summarize(data, predictions)
            stored_predictions[split] = {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in data.items() if key != "features"
            }
            stored_predictions[split]["predictions"] = {
                key: value.cpu() for key, value in predictions.items()
            }
    selection = {"ridge_alpha": chosen_alpha, "ridge_search": alpha_scores,
                 "mlp_epochs": best_epoch, "mlp_tune_mse": best_loss}
    torch.save({"scaler": {key: value.cpu() for key, value in scaler.items()},
                "ridge_weights": ridge_weights.cpu(),
                "read_position": {key: value.cpu() for key, value in read_position.items()},
                "mlp_state_dict": {key: value.cpu() for key, value in mlp.state_dict().items()},
                "selection": selection, "plan": plan}, output_dir / "probe.pt")
    torch.save(stored_predictions, output_dir / "predictions.pt")
    result = {
        "status": "complete", "started_at": plan["created_at"], "finished_at": _now(),
        "elapsed_seconds": time.monotonic() - started, "feature_metadata": metadata,
        "selection": selection, **summaries,
    }
    write_json(output_dir / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    fit(args.feature_dir, args.output_dir, args.device, args.seed)


if __name__ == "__main__":
    main()
