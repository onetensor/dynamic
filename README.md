# Dynamic
## DropSoftmax (hard-drop attention normalization for transformers)

This repo/project is an experiment inspired by DroPE (Sakana AI): use a strong inductive bias during training, then remove it abruptly and observe whether the model rapidly re-stabilizes by internalizing the capability in a different form. This repo focuses on scaffolds of a similar nature, and will be expanded to include refinements on helping models internalize inductive biases of several different mechanisms.

Here, the “scaffold” is softmax attention. We train normally with FlashAttention softmax, then at a configured step we hard-swap the attention operator to a kernel / linear causal attention path (ELU+1 feature map) without resetting optimizer state.

The goal is to study:
- how big the loss spike is,
- how quickly the model recovers,
- and whether the post-drop model exhibits different long-context behavior.

---

## Current progress

### `train_gpt_drop.py`
Adds a DropSoftmax mechanism and a linear attention alternative path.

Key features:
- Hard DropSoftmax switch at `dropsoftmax_step`
  - switches attention implementation from `softmax` → `linear`
  - does not reset optimizer state or schedule
  - emits a clear log marker at the exact flip step
- Segment-aware linear causal attention
  - positive feature map: `phi(x) = elu(x) + 1`
  - fp32 prefix sums for numerical stability
  - supports windowed attention via subtraction keyed off `bm_size`
  - respects document/segment boundaries
- Adds:
  - `attn_impl` flag in attention modules
  - `set_attn_impl()` on attention + GPT wrapper
  - `dropsoftmax_step`, `dropsoftmax_mode` in hyperparameters

Notes:
- The naive prefix-sum implementation can allocate large buffers, which can increase peak memory vs FlashAttention.

---

## Quickstart

### Single run

From the repo root:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt_drop.py
```

## Status / roadmap

Planned next steps:

* Further evaluation of how models respond to the switch to linear attention (early results show that after the switch, both train loss, val loss, and grad norm spike, then recover to around 4.7 train loss after 1.5k steps)
* Chunked windowed linear attention to reduce peak allocation
* Optional single-shot post-drop rescale of linear output RMS (not annealing)
* Deeper evaluation: length sweeps, retrieval-style long-context probes
* Improvements for how models integrate inductive biases such as DropSoftmax and DroPE into weights (work in progress)


```
