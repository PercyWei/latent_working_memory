# latent_working_memory

Research code for streaming mutable latent working memory and matched-budget
context-compression experiments.

The repository keeps paper reproductions separate from new methods. ICAE v1,
the compressor initialization used by C-DIC, is migrated under
`reproductions/icae/` with its own `uv` environment.

## Local development

```bash
uv sync --frozen
uv run pytest
```

## ICAE environment

The ICAE lock targets Linux x86-64. It uses the older PyTorch CUDA 11.8 wheel
that matches ICAE's 2023 dependency stack; the server's CUDA 13.0-capable
driver is backward compatible with that runtime. Project setup does not
download model weights or datasets.

```bash
uv sync --project reproductions/icae --frozen
uv run --project reproductions/icae icae-check-environment
```

See `reproductions/icae/README.md` before running on the GPU server.
