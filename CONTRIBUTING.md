# Contributing

Thanks for helping. This is a Python project; issues and pull requests are welcome.

## Set up and check your change

Needs Python 3.11+ (CI uses 3.12; the recorded run used 3.14). `make setup` installs the CPU build
of PyTorch, which is all the tests need. Caches, downloads and virtual environments stay in
git-ignored folders inside the repo (`.tmp/`, `.venv/`).

```bash
make setup   # CPU PyTorch, the package and dev tools
make lint    # ruff, ruff format --check, mypy --strict
make test    # unit and end-to-end tests on a tiny random model (no GPU, no download)
make audit   # known-vulnerability check (as in CI)
```

The GPU comparison needs an NVIDIA GPU with about 4 GB of free VRAM:

```bash
make setup-gpu   # CUDA 13.0 PyTorch, bitsandbytes, accelerate, nvidia-ml-py
make run         # or make run-lite on smaller machines
```

`make run` deletes the downloaded base model and the trained adapter once the numbers are written;
`make purge` deletes anything left in `.tmp`, including the dataset.

A pull request is ready when `make lint` and `make test` pass and CI is green.

## Guidelines

- Keep each change focused; explain the problem it solves in the pull request.
- Add or update a test for every behaviour change, and update `docs/results/` when numbers change.
- New numbers come from a recorded run (`docs/results/*.json`), never from memory.
- Significant design changes need an ADR in `docs/adr/` (copy `0001` for the format).
- Keep function comments to one short phrase; let names and tests explain the rest.
- Report security problems privately, as described in [SECURITY.md](SECURITY.md).

By contributing you agree that your work is released under the [MIT License](LICENSE).
