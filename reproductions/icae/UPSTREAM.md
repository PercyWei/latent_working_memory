# Upstream provenance and migration record

Migration date: 20260903

## Upstream

- Repository: https://github.com/getao/icae
- Commit: `469a46886a92dd5e76b2d12a8bac0fb7ed7d4cdd`
- Commit date: `2024-05-11T16:53:58-07:00`
- Upstream license: CC0-1.0, copied to `LICENSE`
- Selected implementation: `code/icae_v1`, because C-DIC uses the public
  Llama-2-7B-Chat ICAE checkpoint with 128 compression tokens.

The customized PEFT snapshot under `code/icae_v1/peft` identifies itself as
PEFT `0.4.0.dev0` and retains its Apache-2.0 license in `vendor/peft/LICENSE`.

## Migration mapping

| Upstream path | Local path |
|---|---|
| `code/icae_v1/base` | `src/icae/base` |
| `code/icae_v1/utils` | `src/icae/utils` |
| `code/icae_v1/llama_icae_modeling.py` | `src/icae/llama_icae_modeling.py` |
| `code/icae_v1/llama_icae_learning.py` | `src/icae/llama_icae_learning.py` |
| `code/icae_v1/ft_inference.py` | `examples/ft_inference_upstream.py` |
| `code/icae_v1/peft/src/peft` | `vendor/peft/src/peft` |

The upstream folder `icae_v1` was installed as package `icae` so that its own
absolute imports, such as `from icae.utils import stable_trainer`, resolve
without modifying their meaning.

## Local compatibility changes

1. `optimum.bettertransformer.BetterTransformer` is imported only when the
   optional `better_transformer` flag is enabled. The upstream code imported it
   unconditionally, which made the default path depend on an unpinned optional
   package. The default behavior remains unchanged.
2. Packaging, environment validation, and tests were added outside the migrated
   ICAE implementation.
3. The original inference example is retained unchanged, including its
   placeholder paths; it is not registered as a CLI.
4. The server lock uses PyTorch 2.0.1+cu118. This matches the PyTorch version
   referenced by the vendored PEFT setup more closely than current cu130 builds;
   the CUDA 13.0-capable NVIDIA driver can run the older bundled CUDA runtime.
5. Trailing whitespace was normalized in copied text and Python files; this has
   no runtime effect.

## Known upstream v1 gaps

- `ft_inference.py` contains the misspelled name `memopry_mask`.
- It imports `instruct_ft_tokenize_function`, which is absent from the released
  `llama_icae_learning.py` snapshot.
- File paths and output locations are placeholders rather than CLI arguments.

These issues are documented rather than silently fixed in the retained example.
The project inference runner should implement the required behavior separately
and be validated with the released checkpoint.

No model architecture, compression-token logic, loss, trainer, or customized
PEFT implementation was otherwise changed during migration.
