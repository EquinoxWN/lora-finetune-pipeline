# ADR 0003: A 0.5B base model in 4-bit, a memory preflight, and model files deleted after each run

- **Status:** Accepted

## Context

The comparison runs on a laptop: an RTX 4050 with 6 GB of VRAM and 13.7 GB of RAM, of which only
about 2 to 3 GB is free while other builds run. A larger base model (1.5B or 3B) would prompt better
and make the comparison harder for fine-tuning, which is the more interesting test. But the 1.5B
model's weights file alone is 3.1 GB, more than the free RAM, and the few-shot prompt (77 worked
examples, about 2,000 tokens) needs VRAM for activations on top of the weights. The owner's rule is
also that downloaded and trained model files are deleted once their numbers are recorded.

## Decision

- Use `Qwen/Qwen2.5-0.5B-Instruct` (Apache 2.0) pinned to one commit and to the SHA-256 of its
  988 MB weights file, loaded in 4-bit NF4 with double quantization.
- Before loading, `lfp run` checks free RAM and free VRAM against the config (`min_free_ram_mib`,
  2 GB of VRAM) and exits with code 3 instead of pushing the machine into swap. VRAM is capped at
  90% per process.
- Ship two configs: `m1` (the full held-out set, one epoch) and `lite` (5 held-out messages per
  intent, 150 steps) for machines with less memory or time.
- `lfp run` writes the results JSON first, then deletes the Hugging Face cache and the adapter and
  records every deleted file in the same JSON. `lfp purge --data` deletes the dataset.

## Consequences

- The result is a statement about a 0.5B model. A bigger model would likely narrow the gap between
  prompting and fine-tuning; the config makes that a one-line change on a larger machine.
- Reproducing a number requires downloading the model and data again; the pins make sure it is the
  same model and data, and the split fingerprints in the results show the split is identical.
- Nothing large is left on disk after a run, and nothing can be accidentally committed.
