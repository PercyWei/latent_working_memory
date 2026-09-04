# ICAE v1 reproduction

This directory contains the ICAE v1 code required as the compressor base for
the later C-DIC reproduction. It targets the paper's Llama-2-7B-Chat path, not
the later Mistral-based ICAE v2 release.

No benchmark, dataset, model weight, or checkpoint is stored or downloaded by
the repository setup.

## Source layout

- `src/icae/`: migrated ICAE v1 model and training code.
- `vendor/peft/`: the customized PEFT 0.4.0.dev0 source shipped by ICAE.
- `src/icae_repro/`: local environment checks that do not download models.
- `examples/ft_inference_upstream.py`: the original path-based inference example,
  retained for reference and not used as a production entry point.
- `UPSTREAM.md`: source revision, migration mapping, and local patch record.

## Server environment

Target hardware:

- NVIDIA A800-SXM4-80GB;
- Linux x86-64;
- CUDA 13.0-capable server driver/toolkit environment;
- Python 3.10;
- PyTorch 2.0.1 CUDA 11.8 wheel;
- Transformers 4.31.0 and the ICAE-customized PEFT package.

The CUDA 11.8 wheel is intentional. NVIDIA drivers are backward compatible
with applications built against older CUDA toolkits, and PyTorch 2.0.1 is much
closer to ICAE's original 2023 stack than current CUDA 13.0 PyTorch releases.
No local CUDA extension is compiled by the migrated code.

Create the environment on the server:

```bash
uv sync --project reproductions/icae --frozen
uv run --project reproductions/icae icae-check-environment
```

The environment check reports PyTorch, CUDA, GPU capability, and bfloat16
support without accessing Hugging Face or downloading weights.

The upstream ICAE instructions require bfloat16 rather than fp16 training and
only support training batch size 1 in the released path. Preserve those limits
until a controlled compatibility test justifies changing them.

## Current migration boundary

The migrated model and trainer sources are packaged and syntax-checked, but the
upstream v1 inference example is not an end-to-end CLI. It contains placeholder
paths, references a tokenization helper absent from the released v1 training
file, and contains an upstream variable-name typo. The example is kept unchanged
for provenance. A tested project inference runner will be added only after the
base model and ICAE checkpoint are available on the server.

For a first inference demonstration, prepare a JSONL file with one record per
example:

```json
{"input": "long context", "prompt": "question about the context", "answer": "reference answer"}
```

`answer` is used for comparison and is not supplied to the model during
generation.

## Deferred resources

The following remain server-side setup tasks and are intentionally absent:

- `meta-llama/Llama-2-7b-chat-hf` access and local model path;
- the ICAE Llama-2 checkpoint;
- PwC, MSC, REALTALK, LongMemEval, or other datasets;
- generated predictions, checkpoints, and benchmark outputs.

Do not add these files to Git. The root `.gitignore` excludes common model and
artifact paths and extensions.
