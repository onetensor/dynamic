## The “DropSoftmax” design I recommend (teacherless, hard drop)

### High-level schedule (single model, no teacher/student)

This is the DroPE-style mid-training surgery: take a checkpoint, change a core mechanism, resume training. DroPE explicitly demonstrates this style of “remove and resume” for positional embeddings. ([Sakana][1])

So a “DropSoftmax” experiment is: *train with standard softmax attention (fast convergence), then hard-switch the attention rule to a no-softmax kernel/linear form and keep training*. The loss spike + quick recovery would be the exact phenomenon you’re looking to probe.

**Phase A (baseline):** normal softmax attention (your existing FlashAttn3 path).
**Hard drop moment:** at a chosen global step (I recommend **exactly at a major schedule boundary**, e.g. **2/3 of training steps** to mirror existing window bumps). ([LessWrong][3])
**Phase B:** replace attention with a **no-softmax causal kernel attention** (linear attention) and keep training with the *same* optimizer states + LR schedule (no anneal).

No teacher, no distillation, no annealing—just a hard mechanism swap and continued SGD.

---

## What “no-softmax attention” should be (so it doesn’t explode)

If you literally delete softmax and do `(QKᵀ)V`, you’ll get sign cancellations and scale blowups. The practical “drop softmax” that tends to be trainable is **kernelized linear attention**:

[
A(Q,K,V)*t \approx \frac{\phi(Q_t)\left(\sum*{i\le t}\phi(K_i)V_i^\top\right)}{\phi(Q_t)\left(\sum_{i\le t}\phi(K_i)\right) + \epsilon}
]

Where:

* `φ(x)` is a **positive feature map** (common: `elu(x)+1`)
* the denominator is *not softmax*, it’s just a normalization to keep magnitudes sane
* computation is **O(T d²)** per head rather than O(T²), and is causal via prefix sums

This is “drop softmax” in the sense that the defining probability-simplex projection is gone; the model must internalize stable attention behaviors in its weights after the switch.

---

## Spec: concrete implementation steps for modded-nanogpt #31


---

### 1) Add one new training argument: `--dropsoftmax_step`

Add CLI/config support:

* `dropsoftmax_step: int` (default `-1` meaning disabled)
* `dropsoftmax_mode: str` (optional: `"linear"` only for now)

In the training loop, you want:

* if `step == dropsoftmax_step`:

  * switch a flag on the model: `model.set_attn_impl("linear")`
  * log a big marker: `"=== HARD DROP SOFTMAX NOW ==="`
  * **do not** reset optimizer state, LR schedule, scalars, etc.

Why: DroPE’s story is “the model quickly re-finds a good basin with the scaffold removed” — you want the same conditions. ([Sakana][1])

---

### 2) Add a second attention path in the attention module

Wherever #31 defines attention (likely a `CausalSelfAttention` / `Attention` module), it currently does roughly:

* compute Q,K,V
* apply RoPE (+ dynamic YaRN changes)
* call FlashAttn3 varlen kernel with `causal=True` and `window_size=(..., 0)` ([LessWrong][3])

You’ll add:

* `self.attn_impl = "softmax"` (default)
* a branch in `forward()`:

#### 2A) Softmax path (unchanged)

Keep the FlashAttn3 path exactly as-is for Phase A.

#### 2B) Linear/kernal path (new)

Implement a function like:

```python
def linear_causal_attention(q, k, v, *, window_tokens=None, eps=1e-6):
    # q,k,v: (B, T, H, D)
    # returns: (B, T, H, Dv) typically Dv==D

    # 1) feature map (positive)
    q = F.elu(q) + 1
    k = F.elu(k) + 1

    # 2) prefix sums for causal attention
    # kv: (B,T,H,D,Dv)
    kv = torch.einsum('bthd,bthm->bthdm', k, v)

    kv_cum = kv.cumsum(dim=1)
    k_cum  = k.cumsum(dim=1)

    if window_tokens is not None:
        # sliding window version: subtract prefix at t-window
        kv_cum = kv_cum - shift_right(kv_cum, window_tokens)
        k_cum  = k_cum  - shift_right(k_cum,  window_tokens)

    # 3) normalize
    num = torch.einsum('bthd,bthdm->bthm', q, kv_cum)
    den = torch.einsum('bthd,bthd->bth',  q, k_cum).unsqueeze(-1).clamp_min(eps)
    return num / den
```

**Important engineering choices (strongly recommended):**

* Do `cumsum` in **fp32** even if activations are bf16: it reduces NaNs at essentially zero conceptual cost.
* Keep whatever **QK-norm** / scaling #31 already uses; it’s usually stabilizing before the feature map.

---

### 3) Preserve “long-short window” behavior (optional but recommended)

Around this timeframe, the setup uses a layerwise pattern like:

`[short, short, short, long, short, short, short, short, short, long]` (10 layers) ([LessWrong][3])

…and increases those windows during training (128/384 → 384/896 → 640/1408 …) with YaRN being applied at these changes. ([LessWrong][3])

To keep your experiment comparable, **feed the same per-layer window size into `linear_causal_attention(..., window_tokens=...)`**.

Concretely:

* wherever the model currently computes `window_size` (for FlashAttn3’s `window_size=(bm_size,0)`), also compute `window_tokens = window_size_in_tokens`
* pass that into the linear path for Phase B

This gives you: *same receptive field schedule, only the attention normalization rule changes.*

---

### 4) Handle YaRN + RoPE “as-is” (don’t conflate changes)

#31’s signature change is dynamic YaRN integration, and the PR discussion makes it clear it’s doing careful rescaling when window sizes jump. ([GitHub][4])

For your **first** DropSoftmax run:

* keep RoPE + YaRN logic identical in both phases
* only swap the attention aggregation function

(You can later try “DropSoftmax + DroPE” combos, but don’t start there.)

---

### 5) Torch.compile realities (don’t let it derail you)

A runtime branch on `self.attn_impl` will likely cause one of:

* recompilation on the first step after the flip (fine: it happens once), or
* graph breaks (still fine for an experiment)

Given your goal is “observe the loss spike & recovery”, correctness > peak speed.


## Recommended experiment settings (minimal but informative)

### Pick a hard drop point

Use `dropsoftmax_step = int(0.67 * max_steps)`.

Why: it lines up with the existing “big schedule transitions” mindset (window changes, YaRN changes) seen in this training style. ([LessWrong][3])

### Run 2 baselines

1. **Control:** softmax attention all the way
2. **DropSoftmax:** hard switch at 2/3

Log:

* loss every N steps around the switch (dense logging around ±200 steps)
* gradient norm / NaN counters
* optionally: “effective attention entropy” before/after (just for debugging)
