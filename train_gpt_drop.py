import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
import math

from dataclasses import dataclass, fields
from functools import lru_cache
from itertools import accumulate
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
import numpy as np
import triton
import triton.language as tl
try:
    from flash_attn_interface import flash_attn_varlen_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
    except ImportError as exc:
        raise ImportError(
            "flash_attn_varlen_func not found. Install flash-attn or ensure flash_attn_interface is on PYTHONPATH."
        ) from exc
import torch._dynamo as dynamo
dynamo.config.recompile_limit = 64
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99

def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]

@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ns_line_1_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ns_line_1(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ns_line_1_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
    )
    return out

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ns_line_2_kernel(
    A_ptr, C_ptr,
    M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from ns_line_1_kernel, but also loads and adds a block of A
    # Performance is slightly slower than ns_line_1_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ns_line_2(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ns_line_2_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
    )
    return out

@torch.compile(dynamic=False, fullgraph=True) # Must use dynamic=False or else it's much slower
def newton_schulz_triton(G: torch.Tensor):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    ns_line_3 = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the NS iterations
    for _ in range(5):
        ns_line_1(X, out=A)  # A = X @ X.mT
        ns_line_2(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        ns_line_3(X, B, X, beta=a, out=C)  # C = a * X + B @ X
        X, C = C, X  # Swap references to avoid unnecessary copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# -----------------------------------------------------------------------------
# Muon optimizer

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, and @vagrawal.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_gather_futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            grad_pad = [param.grad for param in params] + [torch.zeros_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                # This gives strange dynamo warnings
                reduce_scatter_futures.append(dist.reduce_scatter(grad, grad_pad[base_i:base_i + world_size], op=dist.ReduceOp.AVG, async_op=True).get_future())

        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            momentum = group["momentum"]
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    grad = p.grad
                    eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                    eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    p.mul_(1 - eff_weight_decay)
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    grad = grad.lerp_(momentum_buffer, momentum)
                    v = newton_schulz_triton(grad)
                    p.add_(other=v, alpha=-eff_lr)
                idx += 1
                all_gather_futures.append(dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank], async_op=True).get_future())
        torch.futures.collect_all(all_gather_futures).wait()

class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # DistributedAdam implementation by @vagrawal

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_gather_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_gather_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_gather_futures).wait()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = cos[None, :x_BTHD.size(-3), None, :], sin[None, :x_BTHD.size(-3), None, :]
    x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3).type_as(x_BTHD)

def _sample_tensor(x: Tensor, max_samples: int):
    if max_samples <= 0:
        return None
    x_flat = x.reshape(-1)
    if x_flat.numel() == 0:
        return None
    if x_flat.numel() <= max_samples:
        return x_flat.detach()
    idx = torch.randperm(x_flat.numel(), device=x_flat.device)[:max_samples]
    return x_flat[idx].detach()

class LinStatsAccumulator:
    def __init__(self, *, eps: float, sample_size: int, collect_hist: bool):
        self.eps = eps
        self.sample_size = sample_size
        self.collect_hist = collect_hist
        self.den_min = float("inf")
        self.den_sum = 0.0
        self.den_count = 0
        self.den_clamp_count = 0
        self.S_norm_max = 0.0
        self.Z_norm_max = 0.0
        self.y_norm_max = 0.0
        self.nan_inf_count = 0
        self.den_samples: list[Tensor] = []
        self.y_norm_samples: list[Tensor] = []

    def update(self, den: Tensor, k_sum: Tensor, kv_sum: Tensor, y: Tensor):
        den_flat = den.reshape(-1)
        self.den_min = min(self.den_min, den_flat.min().item())
        self.den_sum += den_flat.sum().item()
        self.den_count += den_flat.numel()
        self.den_clamp_count += (den_flat < self.eps).sum().item()
        self.nan_inf_count += (~torch.isfinite(den_flat)).sum().item()
        self.nan_inf_count += (~torch.isfinite(y)).sum().item()

        z_norm = k_sum.float().norm(dim=-1).max().item()
        s_norm = kv_sum.float().pow(2).sum(dim=(-1, -2)).sqrt().max().item()
        y_norm = y.float().norm(dim=-1)
        y_norm_max = y_norm.max().item()
        self.Z_norm_max = max(self.Z_norm_max, z_norm)
        self.S_norm_max = max(self.S_norm_max, s_norm)
        self.y_norm_max = max(self.y_norm_max, y_norm_max)

        if self.sample_size > 0:
            den_sample = _sample_tensor(den_flat, self.sample_size)
            if den_sample is not None:
                self.den_samples.append(den_sample.cpu())
            if self.collect_hist:
                y_sample = _sample_tensor(y_norm.reshape(-1), self.sample_size)
                if y_sample is not None:
                    self.y_norm_samples.append(y_sample.cpu())

    def finalize(self):
        den_samples = torch.cat(self.den_samples) if self.den_samples else None
        y_norm_samples = torch.cat(self.y_norm_samples) if self.y_norm_samples else None
        return {
            "den_min": 0.0 if self.den_count == 0 else self.den_min,
            "den_sum": self.den_sum,
            "den_count": self.den_count,
            "den_clamp_count": self.den_clamp_count,
            "S_norm_max": self.S_norm_max,
            "Z_norm_max": self.Z_norm_max,
            "y_norm_max": self.y_norm_max,
            "nan_inf_count": self.nan_inf_count,
            "den_samples": den_samples,
            "y_norm_samples": y_norm_samples,
        }

class LinStatsCollector:
    def __init__(self, *, sample_size: int, collect_hist: bool):
        self.sample_size = sample_size
        self.collect_hist = collect_hist
        self.layer_stats: dict[int, dict] = {}
        self.den_samples: list[Tensor] = []
        self.y_norm_samples: list[Tensor] = []

    def add_layer(self, layer_idx: int, stats: dict):
        self.layer_stats[layer_idx] = stats
        den_samples = stats.get("den_samples")
        y_samples = stats.get("y_norm_samples")
        if den_samples is not None:
            self.den_samples.append(den_samples)
        if y_samples is not None:
            self.y_norm_samples.append(y_samples)

    def _merge_samples(self, samples: list[Tensor]):
        if not samples:
            return None
        merged = torch.cat(samples)
        if merged.numel() > self.sample_size:
            idx = torch.randperm(merged.numel())[: self.sample_size]
            merged = merged[idx]
        return merged

    def aggregate(self):
        if not self.layer_stats:
            return {
                "den_min": 0.0,
                "den_mean": 0.0,
                "den_p01": 0.0,
                "den_clamp_frac": 0.0,
                "S_norm_max": 0.0,
                "Z_norm_max": 0.0,
                "y_norm_max": 0.0,
                "nan_inf_count": 0.0,
                "den_samples": None,
                "y_norm_samples": None,
            }

        den_min = min(s["den_min"] for s in self.layer_stats.values())
        den_sum = sum(s["den_sum"] for s in self.layer_stats.values())
        den_count = sum(s["den_count"] for s in self.layer_stats.values())
        den_clamp_count = sum(s["den_clamp_count"] for s in self.layer_stats.values())
        s_norm_max = max(s["S_norm_max"] for s in self.layer_stats.values())
        z_norm_max = max(s["Z_norm_max"] for s in self.layer_stats.values())
        y_norm_max = max(s["y_norm_max"] for s in self.layer_stats.values())
        nan_inf_count = sum(s["nan_inf_count"] for s in self.layer_stats.values())

        den_samples = self._merge_samples(self.den_samples)
        y_norm_samples = self._merge_samples(self.y_norm_samples)
        den_p01 = 0.0
        if den_samples is not None and den_samples.numel() > 0:
            den_p01 = torch.quantile(den_samples, 0.01).item()

        den_mean = 0.0
        den_clamp_frac = 0.0
        if den_count > 0:
            den_mean = den_sum / den_count
            den_clamp_frac = den_clamp_count / den_count

        return {
            "den_min": den_min,
            "den_mean": den_mean,
            "den_p01": den_p01,
            "den_clamp_frac": den_clamp_frac,
            "S_norm_max": s_norm_max,
            "Z_norm_max": z_norm_max,
            "y_norm_max": y_norm_max,
            "nan_inf_count": nan_inf_count,
            "den_samples": den_samples,
            "y_norm_samples": y_norm_samples,
        }

def _segment_positions(seqlens: Tensor, total_len: int):
    seqlens = seqlens.to(dtype=torch.int64)
    positions = torch.arange(total_len, device=seqlens.device, dtype=torch.int64)
    segment_id = torch.bucketize(positions, seqlens[1:], right=False)
    segment_start = seqlens[segment_id]
    return positions, segment_start

def _segment_cumsum(
    x: Tensor,
    positions: Tensor,
    segment_start: Tensor,
    *,
    window_tokens: int | None = None,
):
    x_cum = torch.cumsum(x, dim=0, dtype=torch.float32)
    zero = x_cum.new_zeros((1,) + x_cum.shape[1:])
    x_cum_pad = torch.cat([zero, x_cum], dim=0)
    offsets = x_cum_pad[segment_start]
    x_cum -= offsets
    if window_tokens is not None:
        shift_pos = positions - segment_start - (window_tokens + 1)
        mask = shift_pos >= 0
        shift_idx = torch.where(mask, segment_start + shift_pos, torch.zeros_like(shift_pos))
        shifted = x_cum[shift_idx]
        if x_cum.ndim > 1:
            mask = mask.view(-1, *([1] * (x_cum.ndim - 1)))
        shifted = shifted * mask
        x_cum -= shifted
    return x_cum

def _linear_causal_attention_full(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    seqlens: Tensor,
    attn_scale: float,
    window_tokens: int | None = None,
    eps: float = 1e-6,
    stats_accum: LinStatsAccumulator | None = None,
):
    positions, segment_start = _segment_positions(seqlens, q.size(0))
    q = q * attn_scale
    q = F.elu(q) + 1
    k = F.elu(k) + 1
    q_fp32 = q.float()
    k_cum = _segment_cumsum(k, positions, segment_start, window_tokens=window_tokens)
    kv = torch.einsum("thd,thm->thdm", k, v)
    kv_cum = _segment_cumsum(kv, positions, segment_start, window_tokens=window_tokens)
    num = torch.einsum("thd,thdm->thm", q_fp32, kv_cum)
    den = torch.einsum("thd,thd->th", q_fp32, k_cum).unsqueeze(-1)
    y = (num / den.clamp_min(eps)).to(dtype=q.dtype)
    if stats_accum is not None:
        stats_accum.update(den.squeeze(-1), k_cum, kv_cum, y)
    return y

def _iter_segments(seqlens: Tensor, total_len: int):
    seqlens = seqlens.to(dtype=torch.int64)
    if seqlens.is_cuda:
        seqlens = seqlens.cpu()
    bounds = seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        if start >= total_len:
            break
        if end <= start:
            continue
        if end > total_len:
            end = total_len
        yield start, end

def _linear_attention_chunked_no_window(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    attn_scale: float,
    chunk_size: int,
    eps: float,
    stats_accum: LinStatsAccumulator | None = None,
):
    out = torch.empty_like(v)
    k_prefix = torch.zeros((q.size(1), q.size(2)), dtype=torch.float32, device=q.device)
    kv_prefix = torch.zeros((q.size(1), q.size(2), v.size(2)), dtype=torch.float32, device=q.device)
    for s in range(0, q.size(0), chunk_size):
        e = min(s + chunk_size, q.size(0))
        q_chunk = q[s:e] * attn_scale
        k_chunk = k[s:e]
        v_chunk = v[s:e]
        q_feat = F.elu(q_chunk) + 1
        k_feat = F.elu(k_chunk) + 1
        q_fp32 = q_feat.float()
        k_cum = torch.cumsum(k_feat, dim=0, dtype=torch.float32) + k_prefix
        kv = torch.einsum("thd,thm->thdm", k_feat, v_chunk)
        kv_cum = torch.cumsum(kv, dim=0, dtype=torch.float32) + kv_prefix
        num = torch.einsum("thd,thdm->thm", q_fp32, kv_cum)
        den = torch.einsum("thd,thd->th", q_fp32, k_cum).unsqueeze(-1)
        y = (num / den.clamp_min(eps)).to(dtype=q.dtype)
        out[s:e] = y
        if stats_accum is not None:
            stats_accum.update(den.squeeze(-1), k_cum, kv_cum, y)
        k_prefix = k_cum[-1]
        kv_prefix = kv_cum[-1]
    return out

def _linear_attention_chunked_window(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    attn_scale: float,
    window_tokens: int,
    chunk_size: int,
    eps: float,
    stats_accum: LinStatsAccumulator | None = None,
):
    total_len = q.size(0)
    num_chunks = (total_len + chunk_size - 1) // chunk_size
    # Prefix sums over chunks to avoid full-length kv_cum.
    k_prefix = torch.zeros((num_chunks + 1, q.size(1), q.size(2)), dtype=torch.float32, device=q.device)
    kv_prefix = torch.zeros((num_chunks + 1, q.size(1), q.size(2), v.size(2)), dtype=torch.float32, device=q.device)
    for c in range(num_chunks):
        s = c * chunk_size
        e = min(s + chunk_size, total_len)
        k_chunk = k[s:e]
        v_chunk = v[s:e]
        k_feat = F.elu(k_chunk) + 1
        k_sum = k_feat.sum(dim=0, dtype=torch.float32)
        kv_sum = torch.einsum("thd,thm->hdm", k_feat, v_chunk).to(dtype=torch.float32)
        k_prefix[c + 1] = k_prefix[c] + k_sum
        kv_prefix[c + 1] = kv_prefix[c] + kv_sum

    out = torch.empty_like(v)
    idx_base = torch.arange(chunk_size, device=q.device)
    for c in range(num_chunks):
        s = c * chunk_size
        e = min(s + chunk_size, total_len)
        lc = e - s
        idx = idx_base[:lc] + s
        s_idx = idx - window_tokens + 1
        s_idx = torch.clamp(s_idx, min=0)
        cs = torch.div(s_idx, chunk_size, rounding_mode="floor")
        os = s_idx - cs * chunk_size

        q_chunk = q[s:e] * attn_scale
        k_chunk = k[s:e]
        v_chunk = v[s:e]
        q_feat = F.elu(q_chunk) + 1
        k_feat = F.elu(k_chunk) + 1
        q_fp32 = q_feat.float()
        k_cum = torch.cumsum(k_feat, dim=0, dtype=torch.float32)
        kv = torch.einsum("thd,thm->thdm", k_feat, v_chunk)
        kv_cum = torch.cumsum(kv, dim=0, dtype=torch.float32)

        os_minus1 = (os - 1).clamp(min=0)
        k_prev = k_cum[os_minus1]
        kv_prev = kv_cum[os_minus1]
        os_is_zero = os == 0
        if os_is_zero.any():
            k_prev = k_prev * (~os_is_zero).view(-1, 1, 1)
            kv_prev = kv_prev * (~os_is_zero).view(-1, 1, 1, 1)
        sum_k_same = k_cum - k_prev
        sum_kv_same = kv_cum - kv_prev

        k_full = k_prefix[c] - k_prefix[cs + 1]
        kv_full = kv_prefix[c] - kv_prefix[cs + 1]

        partial_k = torch.zeros_like(sum_k_same)
        partial_kv = torch.zeros_like(sum_kv_same)
        if c > 0:
            unique_cs = torch.unique(cs)
            for u in unique_cs.tolist():
                if u >= c:
                    continue
                mask_u = cs == u
                if not mask_u.any():
                    continue
                us = u * chunk_size
                ue = min(us + chunk_size, total_len)
                k_u = k[us:ue]
                v_u = v[us:ue]
                k_u_feat = F.elu(k_u) + 1
                k_u_cum = torch.cumsum(k_u_feat, dim=0, dtype=torch.float32)
                kv_u = torch.einsum("thd,thm->thdm", k_u_feat, v_u)
                kv_u_cum = torch.cumsum(kv_u, dim=0, dtype=torch.float32)

                os_u = os[mask_u]
                os_u_minus1 = (os_u - 1).clamp(min=0)
                k_u_prev = k_u_cum[os_u_minus1]
                kv_u_prev = kv_u_cum[os_u_minus1]
                os_u_zero = os_u == 0
                if os_u_zero.any():
                    k_u_prev = k_u_prev * (~os_u_zero).view(-1, 1, 1)
                    kv_u_prev = kv_u_prev * (~os_u_zero).view(-1, 1, 1, 1)
                k_chunk_sum = k_prefix[u + 1] - k_prefix[u]
                kv_chunk_sum = kv_prefix[u + 1] - kv_prefix[u]
                partial_k[mask_u] = k_chunk_sum - k_u_prev
                partial_kv[mask_u] = kv_chunk_sum - kv_u_prev

        sum_k_other = k_cum + k_full + partial_k
        sum_kv_other = kv_cum + kv_full + partial_kv

        same_mask = cs == c
        sum_k = torch.where(same_mask.view(-1, 1, 1), sum_k_same, sum_k_other)
        sum_kv = torch.where(same_mask.view(-1, 1, 1, 1), sum_kv_same, sum_kv_other)

        num = torch.einsum("thd,thdm->thm", q_fp32, sum_kv)
        den = torch.einsum("thd,thd->th", q_fp32, sum_k).unsqueeze(-1)
        y = (num / den.clamp_min(eps)).to(dtype=q.dtype)
        out[s:e] = y
        if stats_accum is not None:
            stats_accum.update(den.squeeze(-1), sum_k, sum_kv, y)
    return out

def linear_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    seqlens: Tensor,
    attn_scale: float,
    window_tokens: int | None = None,
    eps: float = 1e-6,
    chunk_size: int | None = None,
    stats_accum: LinStatsAccumulator | None = None,
):
    if chunk_size is None or chunk_size <= 0:
        return _linear_causal_attention_full(
            q,
            k,
            v,
            seqlens=seqlens,
            attn_scale=attn_scale,
            window_tokens=window_tokens,
            eps=eps,
            stats_accum=stats_accum,
        )
    out = torch.empty_like(v)
    for start, end in _iter_segments(seqlens, q.size(0)):
        q_seg = q[start:end]
        k_seg = k[start:end]
        v_seg = v[start:end]
        if window_tokens is None:
            out[start:end] = _linear_attention_chunked_no_window(
                q_seg,
                k_seg,
                v_seg,
                attn_scale=attn_scale,
                chunk_size=chunk_size,
                eps=eps,
                stats_accum=stats_accum,
            )
        else:
            out[start:end] = _linear_attention_chunked_window(
                q_seg,
                k_seg,
                v_seg,
                attn_scale=attn_scale,
                window_tokens=window_tokens,
                chunk_size=chunk_size,
                eps=eps,
                stats_accum=stats_accum,
            )
    return out

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    rotary_cos: torch.Tensor
    rotary_sin: torch.Tensor
    attn_scale: float
    lin_stats: LinStatsCollector | None = None

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        hdim = num_heads * head_dim
        assert hdim == dim, "num_heads * head_dim must equal model_dim"
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkvo_w = nn.Parameter(torch.empty(4, hdim, dim))
        with torch.no_grad():
            self.qkvo_w[:3].uniform_(-bound, bound) # init QKV weights
            self.qkvo_w[3].zero_() # init output weights to zero

        # sparse gated attention to enable context based no-op by @classiclarryd
        self.attn_gate = CastedLinear(12, num_heads)
        self.attn_gate.weight.detach().zero_()
        self.attn_impl = "softmax"

    def set_attn_impl(self, attn_impl: str):
        if attn_impl not in ("softmax", "linear"):
            raise ValueError(f"Unsupported attn_impl: {attn_impl}")
        self.attn_impl = attn_impl

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        rotary_cos, rotary_sin = attn_args.rotary_cos, attn_args.rotary_sin
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        seqlens, attn_scale, bm_size = attn_args.seqlens, attn_args.attn_scale, attn_args.bm_size

        q, k, v = F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = rotary(q, rotary_cos, rotary_sin), rotary(k, rotary_cos, rotary_sin)
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v) # @ KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = sa_lambdas[0] * v

        if self.attn_impl == "softmax":
            max_len = args.train_max_seq_len if self.training else (args.val_batch_size // (grad_accum_steps * world_size))
            # use flash_attn over flex_attn @varunneal. flash_attn_varlen suggested by @YouJiacheng
            y = flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
                                       max_seqlen_q=max_len, max_seqlen_k=max_len,
                                       causal=True, softmax_scale=attn_scale, window_size=(bm_size, 0))
            y = y.view(B, T, self.num_heads, self.head_dim)
        elif self.attn_impl == "linear":
            stats_accum = None
            if attn_args.lin_stats is not None:
                stats_accum = LinStatsAccumulator(
                    eps=1e-6,
                    sample_size=attn_args.lin_stats.sample_size,
                    collect_hist=attn_args.lin_stats.collect_hist,
                )
            y = linear_causal_attention(
                q[0], k[0], v[0],
                seqlens=seqlens,
                attn_scale=attn_scale,
                window_tokens=bm_size,
                chunk_size=args.linear_attn_chunk_size,
                stats_accum=stats_accum,
            ).view(B, T, self.num_heads, self.head_dim)
            if stats_accum is not None:
                attn_args.lin_stats.add_layer(self.layer_idx, stats_accum.finalize())
        else:
            raise ValueError(f"Unsupported attn_impl: {self.attn_impl}")
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, self.qkvo_w[3].type_as(y))
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        # make both matrices have the same shape because optimizer sorts params by shape
        # 2 matrices x 12 layers = 24 total, which is divisible by 8 GPU world size
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = F.linear(x, self.c_fc.T.type_as(x))
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.linear(x, self.c_proj.type_as(x))
        return x

class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, head_dim, num_heads, layer_idx) if layer_idx != 7 else None
        # skip MLP blocks for first MLP layer by @EmelyanenkoK
        self.mlp = MLP(dim) if layer_idx != 0 else None

    def forward(self, x: Tensor, x0: Tensor, lambdas: Tensor, attn_args: AttnArgs):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args)
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        use_fp8 = not os.environ.get("DISABLE_FP8", False)
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=use_fp8, x_s=(model_dim**0.5)/448, w_s=2**-9, grad_s=1/448)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        pad = (-num_layers * 5) % dist.get_world_size()
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers), # skip_weights
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)], # block lambdas
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)], # SA lambdas
            torch.ones(pad),
        ]))
        self.max_seq_len = max_seq_len
        self.setup_yarn(head_dim)
        # set learning rates
        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 1.0
        self.scalars.lr_mul = 5.0

    def setup_yarn(self, head_dim: int):
        # store single copy of rotary tensors
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=head_dim//4, dtype=torch.float32)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(head_dim//4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)
        self.rotary_cos = nn.Buffer(theta.cos(), persistent=False)
        self.rotary_sin = nn.Buffer(theta.sin(), persistent=False)
        self.angular_freq = angular_freq

        # scale attention factor f in attn=softmax(f*qk) logarithmically with window size @classiclarryd
        windows = list(dict.fromkeys(list(args.ws_schedule) + [args.ws_validate]))
        scale_factors = [0.2 * math.log(curr / prev) + 1 for prev, curr in zip(windows[:-1], windows[1:])]
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        attn_scales = list(accumulate([0.1] + scale_factors, lambda acc, factor: acc * factor))
        self.attn_scales = dict(zip(windows, attn_scales))

    def set_attn_impl(self, attn_impl: str):
        for block in self.blocks:
            if block.attn is not None:
                block.attn.set_attn_impl(attn_impl)

    def apply_yarn(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.rotary_cos.copy_(theta.cos())
        self.rotary_sin.copy_(theta.sin())

    def set_angular_freq(self, angular_freq: Tensor):
        self.angular_freq.copy_(angular_freq)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.rotary_cos.copy_(theta.cos())
        self.rotary_sin.copy_(theta.sin())

    def forward(
        self,
        input_seq: Tensor,
        target_seq: Tensor,
        seqlens: Tensor,
        ws: int,
        stats: dict | None = None,
        lin_stats: LinStatsCollector | None = None,
    ):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = ws * args.block_size, (ws // 2) * args.block_size
        bm_sizes = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(bm_sizes) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977

        # U-net design by @brendanh0gan
        skip_connections = []
        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)

        n = len(self.blocks) // 2

        for i in range(len(self.blocks)):
            attn_args = AttnArgs(
                ve=ve[i],
                sa_lambdas=sa_lambdas[i],
                seqlens=seqlens,
                bm_size=bm_sizes[i],
                rotary_cos=self.rotary_cos,
                rotary_sin=self.rotary_sin,
                attn_scale=self.attn_scales[ws],
                lin_stats=lin_stats,
            )
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, x0, lambdas[i], attn_args)
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / 7.5)
        if stats is not None:
            logits_fp32 = logits.float()
            stats["logits_maxabs"] = logits_fp32.abs().max()
            stats["logits_std"] = logits_fp32.std()
            sample_tokens = min(logits_fp32.shape[1], stats.get("logits_sample_tokens", 128))
            logits_sample = logits_fp32[:, :sample_tokens].reshape(-1, logits_fp32.size(-1))
            if logits_sample.numel() > 0:
                probs = torch.softmax(logits_sample, dim=-1)
                stats["entropy"] = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()
                if stats.get("collect_hist", False):
                    stats["logits_samples"] = _sample_tensor(logits_sample, stats.get("hist_sample_size", 10000))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction="sum" if self.training else "mean")
        return loss

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

BOS_ID = 50256

class BOSFinder:
    # Helper for getting sequences that start at the beginning of documents by @varunneal based on work by @classiclarryd
    def __init__(self, tokens: Tensor, world_size: int = 1):
        # Precompute BOS positions once per shard
        self.size = tokens.numel()
        self.bos_idx = (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.i = 0
        self.world_size = world_size

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(f"Insufficient BOS ahead of position {cur}; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur
                idx += 1

            assert cur_len == num_tokens_local + 1
        self.i = idx

        return starts, ends

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True):
    # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    tokens = _load_data_shard(next(file_iter))
    finder = BOSFinder(tokens, world_size=world_size) if align_to_bos else None
    pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)  # median doc length is ~400

        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                tokens = _load_data_shard(next(file_iter))
                finder = BOSFinder(tokens, world_size=world_size)
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens


        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        new_params = yield (
            _inputs.to(device="cuda", dtype=torch.int32, non_blocking=True),
            _targets.to(device="cuda", dtype=torch.int64, non_blocking=True),
            _cum_lengths.to(device="cuda", dtype=torch.int32, non_blocking=True)
        )

        if new_params is not None:
            # makes it possible for generator to receive new (num_tokens, max_seq_len, grad_accum_steps) via .send()
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            num_tokens = new_num_tokens
            max_seq_len = new_max_seq_len
            grad_accum_steps = new_grad_accum_steps 


# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files: str = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files: str = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens: int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_batch_size: int = 2048 * 24 * 8
    train_max_seq_len: int = 128 * 16
    val_batch_size: int = 4 * 64 * 1024 * 8
    # optimization
    num_iterations: int = 1670 # number of iterations to run
    cooldown_frac: int = 0.5 # fraction of training spent cooling down the learning rate
    dropsoftmax_step: int = -1 # global step to hard-switch attention; -1 disables
    dropsoftmax_mode: str = "linear"
    # evaluation and logging
    run_id: str = f"yarn/{uuid.uuid4()}"
    val_loss_every: int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    log_interval: int = 10
    hist_interval: int = 1000
    hist_sample_size: int = 10000
    logits_sample_tokens: int = 128
    wandb_project: str = "project name"
    save_checkpoint: bool = False
    # attention masking
    block_size: int = 128
    ws_schedule: tuple = (3, 7, 11)
    ws_validate: int = 13 # increase final validation ws @classiclarryd
    linear_attn_chunk_size: int = 128

def _parse_env_value(raw: str, default):
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, tuple):
        raw = raw.strip()
        if not raw:
            return default
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            return default
        elem_type = type(default[0]) if default else int
        return tuple(elem_type(p) for p in parts)
    return raw

def apply_env_overrides(args: Hyperparameters):
    for field in fields(args):
        env_key = f"HP_{field.name}".upper()
        if env_key not in os.environ:
            continue
        raw = os.environ[env_key]
        if raw == "":
            continue
        default = getattr(args, field.name)
        setattr(args, field.name, _parse_env_value(raw, default))

args = Hyperparameters()
apply_env_overrides(args)

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert 8 % world_size == 0, "world_size must be a divisor of 8"
grad_accum_steps = 8 // world_size
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = args.run_id
    logfile = f"logs/{run_id}.txt"
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running Triton version {triton.__version__}")

def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

wandb_run = None
if master_process and WANDB_AVAILABLE:
    wandb_run = wandb.init(project=args.wandb_project)
    wandb.config.update({field.name: getattr(args, field.name) for field in fields(args)}, allow_val_change=True)

model: nn.Module = GPT(
    vocab_size=50257,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    model_dim=768,
    max_seq_len=max(args.train_batch_size, args.val_batch_size) // (grad_accum_steps * world_size)
).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s)
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = DistAdam(scalar_params + head_params + embed_params, lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, weight_decay=0.0)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations
    assert 0 <= x < 1
    lr = 1.0
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        lr = w * 1.0 + (1 - w) * 0.1
    return lr

def get_ws(step: int):
    if step == args.num_iterations:
        return args.ws_validate
    x = step / (1 + args.num_iterations)
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx]

def get_ws_phase(step: int):
    if step == args.num_iterations:
        return args.ws_validate, len(args.ws_schedule)
    x = step / (1 + args.num_iterations)
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx], ws_idx

def should_log(step: int):
    if args.dropsoftmax_step >= 0 and abs(step - args.dropsoftmax_step) <= 200:
        return True
    return args.log_interval > 0 and (step % args.log_interval == 0)

def should_log_hist(step: int):
    if args.dropsoftmax_step >= 0:
        for offset in (-20, 20, 100, 500):
            if step == args.dropsoftmax_step + offset:
                return True
    return args.hist_interval > 0 and (step % args.hist_interval == 0)

def compute_grad_norm(params: list[Tensor]):
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += p.grad.float().pow(2).sum().item()
    return math.sqrt(total)

def compute_param_norm(params: list[Tensor]):
    total = 0.0
    for p in params:
        total += p.detach().float().pow(2).sum().item()
    return math.sqrt(total)

def compute_update_norm(params: list[Tensor], snapshot: list[Tensor]):
    total = 0.0
    for p, p_old in zip(params, snapshot):
        diff = (p.detach() - p_old).float()
        total += diff.pow(2).sum().item()
    return math.sqrt(total)

model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)
all_params = list(model.parameters())

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 30
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
train_loader = distributed_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
for step in range(warmup_steps):
    inputs, targets, cum_seqlens = next(train_loader)
    ws = args.ws_schedule[step % len(args.ws_schedule)]  # each window size is a new graph, need to warm up each
    model(inputs, targets, cum_seqlens, ws).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
training_time_ms = 0
# validation helper
def eval_loss(ws_value: int):
    assert args.val_tokens % args.val_batch_size == 0
    val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
    val_loader = distributed_data_generator(
        args.val_files,
        args.val_batch_size,
        -1,
        grad_accum_steps=grad_accum_steps,
        align_to_bos=False,
    )
    val_loss = 0
    with torch.no_grad():
        for _ in range(val_steps):
            inputs, targets, cum_seqlens = next(val_loader)
            val_loss += model(inputs, targets, cum_seqlens, ws_value)
    val_loss /= val_steps
    del val_loader
    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    return val_loss
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
ws, ws_phase = get_ws_phase(0)
current_attn_impl = "softmax"
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    new_ws, new_phase = get_ws_phase(step)
    if new_ws != ws:
        model.apply_yarn(ws, new_ws)
        ws = new_ws
    ws_phase = new_phase
    if args.dropsoftmax_step >= 0 and step == args.dropsoftmax_step:
        model.set_attn_impl(args.dropsoftmax_mode)
        current_attn_impl = args.dropsoftmax_mode
        print0("=== HARD DROP SOFTMAX NOW ===", console=True)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_loss_ctx = eval_loss(ws)
        val_loss_long = val_loss_ctx
        if ws != args.ws_validate:
            angular_backup = model.angular_freq.clone()
            model.apply_yarn(ws, args.ws_validate)
            val_loss_long = eval_loss(args.ws_validate)
            model.set_angular_freq(angular_backup)
        if master_process:
            print0(
                f"step:{step}/{train_steps} "
                f"val_loss:{val_loss_ctx:.4f} val_loss_long:{val_loss_long:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms",
                console=True,
            )
        if master_process and WANDB_AVAILABLE and wandb_run is not None:
            wandb.log(
                {
                    "val/loss_ctx2048": val_loss_ctx.item(),
                    "val/loss_long": val_loss_long.item(),
                },
                step=step,
            )
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    do_log = should_log(step)
    log_hist = should_log_hist(step)
    collect_stats = (do_log or log_hist) and master_process and WANDB_AVAILABLE and (wandb_run is not None)
    stats = None
    lin_stats = None
    if collect_stats:
        stats = {
            "collect_hist": log_hist,
            "hist_sample_size": args.hist_sample_size,
            "logits_sample_tokens": args.logits_sample_tokens,
        }
        if current_attn_impl == "linear":
            lin_stats = LinStatsCollector(sample_size=args.hist_sample_size, collect_hist=log_hist)

    loss_total = torch.zeros((), device=device) if do_log else None
    for micro_idx in range(grad_accum_steps):
        inputs, targets, cum_seqlens = next(train_loader)
        use_stats = collect_stats and (micro_idx == grad_accum_steps - 1)
        loss = model(
            inputs,
            targets,
            cum_seqlens,
            ws,
            stats=stats if use_stats else None,
            lin_stats=lin_stats if use_stats else None,
        )
        loss.backward()
        if do_log:
            loss_total += loss.detach()

    train_loss = None
    if do_log:
        train_loss = loss_total / (args.train_batch_size / world_size)
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)

    param_snapshot = None
    grad_norm = 0.0
    param_norm = 0.0
    update_norm = 0.0
    if do_log and master_process:
        grad_norm = compute_grad_norm(all_params)
        param_snapshot = [p.detach().clone() for p in all_params]
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1) # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        opt.step()
    if do_log and master_process:
        update_norm = compute_update_norm(all_params, param_snapshot)
        param_norm = compute_param_norm(all_params)
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)
    if do_log and master_process and WANDB_AVAILABLE and wandb_run is not None:
        tokens_processed = (step + 1) * args.train_batch_size
        tok_per_s = 0.0
        if approx_training_time_ms > 0:
            tok_per_s = tokens_processed / (approx_training_time_ms / 1000)
        steps_since_drop = step - args.dropsoftmax_step if args.dropsoftmax_step >= 0 else -1
        attn_impl_flag = 1 if current_attn_impl == "linear" else 0
        window_short = (ws // 2) * args.block_size
        window_long = ws * args.block_size

        out_logits_maxabs = 0.0
        out_logits_std = 0.0
        out_entropy = 0.0
        if stats is not None:
            out_logits_maxabs = stats.get("logits_maxabs", torch.tensor(0.0)).item()
            out_logits_std = stats.get("logits_std", torch.tensor(0.0)).item()
            out_entropy = stats.get("entropy", torch.tensor(0.0)).item()

        lin_metrics = {
            "den_min": 0.0,
            "den_p01": 0.0,
            "den_mean": 0.0,
            "den_clamp_frac": 0.0,
            "S_norm_max": 0.0,
            "Z_norm_max": 0.0,
            "y_norm_max": 0.0,
            "nan_inf_count": 0.0,
        }
        lin_den_samples = None
        lin_y_norm_samples = None
        if lin_stats is not None:
            lin_agg = lin_stats.aggregate()
            lin_metrics = {
                "den_min": lin_agg["den_min"],
                "den_p01": lin_agg["den_p01"],
                "den_mean": lin_agg["den_mean"],
                "den_clamp_frac": lin_agg["den_clamp_frac"],
                "S_norm_max": lin_agg["S_norm_max"],
                "Z_norm_max": lin_agg["Z_norm_max"],
                "y_norm_max": lin_agg["y_norm_max"],
                "nan_inf_count": lin_agg["nan_inf_count"],
            }
            lin_den_samples = lin_agg.get("den_samples")
            lin_y_norm_samples = lin_agg.get("y_norm_samples")

        metrics = {
            "train/loss": train_loss.item() if train_loss is not None else 0.0,
            "train/lr": optimizer1.param_groups[0]["lr"],
            "train/step": step,
            "train/tokens": tokens_processed,
            "train/tok_per_s": tok_per_s,
            "train/grad_norm": grad_norm,
            "train/param_norm": param_norm,
            "train/update_norm": update_norm,
            "drop/attn_impl": attn_impl_flag,
            "drop/steps_since_drop": steps_since_drop,
            "drop/drop_step": args.dropsoftmax_step,
            "sched/window_short": window_short,
            "sched/window_long": window_long,
            "sched/yarn_scale": float(model.attn_scales[ws]),
            "sched/phase_id": ws_phase,
            "lin/den_min": lin_metrics["den_min"],
            "lin/den_p01": lin_metrics["den_p01"],
            "lin/den_mean": lin_metrics["den_mean"],
            "lin/den_clamp_frac": lin_metrics["den_clamp_frac"],
            "lin/S_norm_max": lin_metrics["S_norm_max"],
            "lin/Z_norm_max": lin_metrics["Z_norm_max"],
            "lin/y_norm_max": lin_metrics["y_norm_max"],
            "lin/nan_inf_count": lin_metrics["nan_inf_count"],
            "out/logits_maxabs": out_logits_maxabs,
            "out/logits_std": out_logits_std,
            "out/entropy": out_entropy,
        }

        if abs(steps_since_drop) <= 200 and lin_stats is not None and lin_stats.layer_stats:
            layer_ids = sorted(lin_stats.layer_stats.keys())
            picks = [layer_ids[0], layer_ids[len(layer_ids) // 2], layer_ids[-1]]
            for layer_id in picks:
                metrics[f"lin_l{layer_id}/den_min"] = lin_stats.layer_stats[layer_id]["den_min"]

        if log_hist and lin_den_samples is not None:
            metrics["hist/lin_den"] = wandb.Histogram(lin_den_samples.numpy())
        if log_hist and lin_y_norm_samples is not None:
            metrics["hist/lin_y_norm"] = wandb.Histogram(lin_y_norm_samples.numpy())
        if log_hist and stats is not None and stats.get("logits_samples") is not None:
            metrics["hist/logits"] = wandb.Histogram(stats["logits_samples"].cpu().numpy())

        wandb.log(metrics, step=step)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
if master_process and WANDB_AVAILABLE and wandb_run is not None:
    wandb.finish()
dist.destroy_process_group()
