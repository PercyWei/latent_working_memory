from __future__ import annotations

import json

import torch


def collect_environment() -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    result: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
    }
    if not cuda_available:
        return result

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    result.update(
        {
            "device_index": device,
            "device_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    )
    return result


def main() -> None:
    environment = collect_environment()
    print(json.dumps(environment, indent=2, sort_keys=True))
    if not environment["cuda_available"]:
        raise SystemExit("CUDA is unavailable; run the ICAE environment on the GPU server.")


if __name__ == "__main__":
    main()
