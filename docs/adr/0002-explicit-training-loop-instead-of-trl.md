# ADR 0002: An explicit QLoRA training loop instead of TRL's SFTTrainer

- **Status:** Accepted

## Context

The usual way to fine-tune with LoRA on Hugging Face models is TRL's `SFTTrainer` (or the
Transformers `Trainer`). It handles batching, mixed precision, gradient accumulation, logging and
checkpoints. This project's purpose is a comparison whose numbers people can trust, so three
things have to be visible and testable rather than configured: which tokens carry the loss (only
the answer, never the prompt or the intent list), when memory peaks on a 6 GB GPU, and exactly how
long training takes. The machine also has 2 to 3 GB of free system RAM while other builds run, and
TRL brings `datasets`, which keeps its own Arrow cache.

## Decision

Write the loop in `train.py`: build features with the prompt masked to `-100`, right-pad in
`collate`, run AdamW over the LoRA parameters only with linear warmup and decay, clip gradients at
1.0, use bf16 autocast on the GPU, and time the loop with CUDA synchronisation. PEFT is still used
for the LoRA layers and `prepare_model_for_kbit_training`; bitsandbytes for the 4-bit base model.

## Consequences

- The masking, the schedule, reproducibility (two runs with one seed give identical losses) and the
  end-to-end behaviour are unit-tested on a tiny random Qwen2 model on the CPU, with no download.
- Dependencies stay small: no `datasets`, `trl` or tracking client, so there is no extra cache to
  keep inside the repository and less to audit.
- We give up what the trainers provide for free: checkpoint resume, multi-GPU, packing of short
  sequences, and integrations with run trackers. None is needed for a one-epoch run of a few
  minutes on one GPU; run tracking is an M2 item and will be added explicitly (MLflow) rather than
  through a trainer callback.
- If the project grows to multi-GPU or long runs, moving to TRL is a contained change: the
  features, prompts and evaluation do not depend on the loop.
