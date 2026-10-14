# RFC 0001: lora-finetune-pipeline design

- **Status:** Accepted (M1 implemented)
- **Author:** EquinoxWN
- **Created:** 2026

## Problem

Teams that need a language model to do one narrow job (route a support message, tag a ticket,
extract a field) have to decide whether to fine-tune a model or keep prompting a general one.
The decision is usually made on intuition. Prompting costs nothing up front but sends long prompts
(instructions, the list of allowed answers, examples) with every request; fine-tuning costs a
training run but can answer from a short prompt. Which one wins depends on accuracy and on cost per
query, and both can be measured. This project measures them on one public task, on one consumer
GPU, with everything pinned so the numbers can be reproduced.

**Users:** ML and platform engineers deciding how to ship a narrow LLM task, and reviewers who want
the decision backed by a reproducible measurement rather than a claim.

## Critical-to-quality measures (Define)

| CTQ | Measure | Target |
|---|---|---|
| No leakage | Held-out texts that also appear in train or validation, exact or as the same set of words | 0 |
| Fine-tuning pays off in quality | Accuracy of the QLoRA adapter minus the better of zero-shot and few-shot prompting, same held-out set | at least +10 points |
| Answers are usable | Share of answers that name no allowed intent | reported per mode; under 1% for the adapter |
| Fits a 6 GB laptop GPU | Peak reserved VRAM during training and evaluation | under 5.5 GB (the 90% cap) |
| Cheap to train | Wall time of the training run | under 15 minutes |
| Reproducible | Model commit and weights SHA-256, dataset commit and SHA-256, seed, library versions, split fingerprints recorded | all recorded in the results JSON |
| Clean finish | Model, adapter and dataset files left after the run | 0 |
| Tests need no GPU | Unit and end-to-end tests run on CPU with no download | pass in CI |

## Goals

- **M1 (this RFC):**
  - A public task dataset (BANKING77: 77 banking intents, CC BY 4.0) is downloaded from a pinned
    commit, checked against its SHA-256, curated (labels canonicalised, duplicates and
    contradictory copies removed), split into train, validation and test, and checked for leakage.
  - QLoRA trains LoRA adapters on a 4-bit NF4 quantized base model (Qwen2.5-0.5B-Instruct, Apache
    2.0, pinned commit) on one GPU, with gradient checkpointing and a VRAM cap.
  - The same held-out set is scored three ways: the base model zero-shot (instruction and the
    intent list), the base model few-shot (plus one example per intent), and the base model with
    the adapter (short prompt, no list). Accuracy, macro-F1, invalid-answer rate, time per query,
    prompt tokens, peak VRAM and GPU energy are recorded, and a break-even point is computed.
  - After the numbers are written, the base model, the Hugging Face cache and the adapter are deleted.
- **M2:** run tracking (MLflow) with config, metrics and artifacts; more prompting baselines
  (retrieved nearest-neighbour examples).
- **M3:** error analysis that groups the remaining mistakes; serving the merged adapter with vLLM;
  a model card with the costs.

## Non-goals

- A hosted service, or training anything larger than fits a 6 GB consumer GPU.
- Leaderboard-level accuracy on BANKING77; the point is the comparison under equal conditions.
- Cloud price estimates presented as measurements: the cost note uses only measured GPU time and
  energy on this machine.

## Proposed design

```
BANKING77 CSV (pinned commit, SHA-256) ─> parse (bounded) ─> canonical labels ─> de-duplicate
      ─> drop train texts that match test (exact or same words) ─> stratified validation split
      ─> leakage check (must be 0) ─> splits + fingerprints

Qwen2.5-0.5B-Instruct (pinned commit, weights SHA-256) ─> 4-bit NF4, double quant, bf16 compute
      ├─ zero-shot prompt ─────────────┐
      ├─ few-shot prompt (77 examples) ├─> greedy answer ─> strict parser ─> accuracy, macro-F1,
      └─ + LoRA adapter, short prompt ─┘        invalid rate, ms/query, tokens, VRAM, energy
                ^
   QLoRA training: LoRA r=16 on all attention and MLP projections, loss only on the answer tokens
```

- **Data:** the CSVs come from `PolyAI-LDN/task-specific-datasets` at commit `57ec275`, over HTTPS
  only, capped at their pinned size and refused unless the SHA-256 matches; no partial file is
  kept. Rows are bounded (100,000 rows, 1,000 characters). Two labels are spelled
  inconsistently in the source (`Refund_not_showing_up`, `reverted_card_payment?`) and are
  canonicalised. Texts are compared after case folding, accent stripping and punctuation removal;
  copies with the same label are dropped, and texts that carry two different labels are removed
  entirely. Any training text whose normalised form or word set appears in the test file is
  removed, then 10 examples per intent go to validation. The leakage check runs on the final
  splits and fails the run if any count is not zero.
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` at commit `7ae5576`, loaded in 4-bit NF4 with double
  quantization through bitsandbytes and bf16 compute. It fits the 6 GB GPU with room for training
  and for the 2,000-token few-shot prompts, and its weights file (988 MB) can be memory-mapped
  with about 2 GB of free system RAM.
- **Training:** LoRA (r=16, alpha=32, dropout 0.05) on the q, k, v, o, gate, up and down
  projections; one epoch, batch 16, AdamW at 2e-4 with linear warmup and decay, gradient
  checkpointing, and the loss masked to the answer and end-of-turn tokens. The loop is written
  out in `train.py` rather than hidden in a trainer (ADR 0002).
- **Evaluation:** greedy decoding with at most 16 new tokens, batched with left padding. An answer
  counts only if its first line is exactly one allowed intent after trimming case, quotes and
  punctuation; anything else is invalid and wrong.
- **Measurement:** each phase (load, each evaluation, training) records wall time, peak allocated
  and reserved VRAM through `torch.cuda`, and GPU energy through NVML's total-energy counter.
- **Safety and hygiene:** every cache variable (Hugging Face, torch, Triton, CUDA, matplotlib, temp)
  points inside `.tmp/`; the CLI refuses output and delete paths outside the repository's `.tmp`;
  a preflight refuses to load the model when free RAM or VRAM is below the configured floor; model
  files are deleted by the run itself after the results JSON is written.

## Alternatives considered

| Option | Why not (yet) |
|---|---|
| TRL `SFTTrainer` instead of a hand-written loop | Less code, but it hides the loss masking, the schedule and when memory peaks, and it brings `datasets` and its cache into the dependency tree. The explicit loop is about 80 lines, is tested on a tiny model, and is the thing this project explains (ADR 0002). Worth revisiting in M2 when run tracking is added. |
| A larger base model (1.5B to 3B) in 4-bit | Better zero-shot and few-shot quality, so a fairer test of prompting, but the 1.5B weights file is 3.1 GB and the machine has about 2 to 3 GB of free RAM while other builds run; 3B does not leave room for few-shot prompts in 6 GB of VRAM. Kept as a configuration change for a bigger machine. |
| Full fine-tuning or 16-bit LoRA | Does not fit a 6 GB GPU for anything but the smallest models; QLoRA is the method being evaluated. |
| Scoring every label by log-likelihood instead of generating | Removes invalid answers, but costs 77 forward passes per message and is not how a prompted model is used in practice; strict parsing of a generated answer is cheaper and reports invalid answers as a measured defect. |
| AG News, SST-2 or emotion datasets | Fewer classes, so prompting looks better than it would on a real routing task, and their licences are research-only or unclear; BANKING77 is CC BY 4.0 with 77 fine-grained intents. |

## Measurement plan

`lfp run` (via `make run`) writes `docs/results/m1-run.json` with, per mode, accuracy, macro-F1,
invalid rate, ms per query, mean prompt tokens, peak VRAM and GPU energy, plus training steps, the
loss curve, validation loss before and after, trainable parameters, adapter size, versions, pins,
the curation report and the files deleted. `make run-lite` is the same comparison on 5 test
messages per intent and 150 training steps for machines with less memory or time. CI runs the
unit tests and an end-to-end run of the same pipeline on a tiny random model (no GPU).

## Milestones

- **M1:** a task dataset is curated, de-duplicated and split with no leakage. QLoRA trains small adapters on a quantized base model on a single GPU. The adapter is measured against zero-shot and few-shot prompting on the same held-out set.
- **M2:** every run is tracked with its config, metrics and artifacts; more prompting baselines.
- **M3:** error analysis groups the remaining mistakes; the adapter is served with vLLM and documented in a model card, including costs.

## Risks and open questions

- One task and one small model: the result says when fine-tuning pays off for a 77-way intent
  task on a 0.5B model, not for every task. Larger base models prompt better.
- GPU energy comes from NVML's counter for the whole GPU; other processes using the GPU at the same
  time would be counted. The runs were made with nothing else on the GPU.
- Laptop GPU clocks vary with temperature and power mode, so times can differ between runs by a
  few percent.
