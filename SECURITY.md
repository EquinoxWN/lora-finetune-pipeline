# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub:
**Security > Report a vulnerability** on this repository
(`https://github.com/EquinoxWN/lora-finetune-pipeline/security/advisories/new`). Do not open a public issue.

Include what you found, how to reproduce it, and the impact you expect. You will get an answer
within 7 days. Fixes are released as soon as they are ready, and you are credited unless you ask
not to be.

## Supported versions

This project is pre-1.0. Only the latest commit on `main` receives fixes.

## Scope

- **In scope:** a tampered dataset or model file being used (the dataset is pinned by commit and SHA-256, the model by commit and weights SHA-256), a download over plain HTTP or without a size limit, a config file that makes the tool read or delete files outside the repository's `.tmp` folder, and held-out examples leaking into training.
- **Out of scope:** the quality of the model's answers on your own data; that is a measurement, not a vulnerability.

## How this repository protects itself

- Every GitHub Action is pinned to a full commit SHA, and workflows run with read-only
  permissions and without persisted credentials.
- Dependabot update pull requests are paused (`open-pull-requests-limit: 0`) so the repository keeps a single `main` branch; dependencies are updated by hand and checked by the CI audit.
- CI runs lint, strict type checks, the tests and a known-vulnerability check (pip-audit) on every
  push and pull request.
- Model weights are loaded only from `safetensors` files (no pickle), and adapters are saved in the
  same format.
