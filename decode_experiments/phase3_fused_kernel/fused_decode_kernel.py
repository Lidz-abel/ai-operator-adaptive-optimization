# decode_experiments/phase3_fused_kernel/fused_decode_kernel_v2.py

import math
import time
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
import triton
import triton.language as tl

# ============================================================================
# Optimized Triton Kernel (V2)
# ============================================================================

@triton.jit
def fused_h2o_decode_attention_v2_kernel(
    # 指针
    Q_ptr, K_ptr, V_ptr, Out_ptr, H2O_Score_ptr,
    # 维度
    batch_size, num_heads, head_dim, kv_len,
    # 步长 (Strides)
    stride_q_batch, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_seq, stride_k_head, stride_k_dim,
    stride_v_batch, stride_v_seq, stride_v_head, stride_v_dim,
    stride_o_batch, stride_o_head, stride_o_dim,
    stride_h2o_batch, stride_h2o_head, stride_h2o_seq,
    # 标量
    scale,
    # 编译时常量
    BLOCK_SIZE_KV: tl.constexpr,
    BLOCK_SIZE_HEAD_DIM: tl.constexpr,
):
    # Program ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    # 1. 指针偏移计算
    offset_q = pid_batch * stride_q_batch + pid_head * stride_q_head
    offset_k = pid_batch * stride_k_batch + pid_head * stride_k_head
    offset_v = pid_batch * stride_v_batch + pid_head * stride_v_head
    offset_o = pid_batch * stride_o_batch + pid_head * stride_o_head
    offset_h2o = pid_batch * stride_h2o_batch + pid_head * stride_h2o_head
    
    # 2. 准备维度掩码
    offs_d = tl.arange(0, BLOCK_SIZE_HEAD_DIM)
    mask_d = offs_d < head_dim
    
    # 3. 加载 Q (寄存器常驻)
    q_ptrs = Q_ptr + offset_q + offs_d * stride_q_dim
    q = tl.load(q_ptrs, mask=mask_d, other=0.0)
    
    # 4. 初始化 Online Softmax 累加器
    m_i = -float('inf')  # 局部最大值
    l_i = 0.0            # 局部指数和
    acc = tl.zeros([BLOCK_SIZE_HEAD_DIM], dtype=tl.float32) 
    
    # --- Pass 1: 计算 Attention 并暂存中间分数 ---
    for start_n in range(0, kv_len, BLOCK_SIZE_KV):
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_KV)
        mask_n = offs_n < kv_len
        
        # 加载 K 块 [BLOCK_SIZE_KV, D]
        k_ptrs = K_ptr + offset_k + offs_n[:, None] * stride_k_seq + offs_d[None, :] * stride_k_dim
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        # 计算 QK^T [BLOCK_SIZE_KV]
        qk = tl.sum(q[None, :] * k, axis=1) * scale
        qk = tl.where(mask_n, qk, -float('inf'))
        
        # 暂时写回 raw_qk 以便后续归一化 (仅 1D 写入，开销极小)
        h2o_ptrs = H2O_Score_ptr + offset_h2o + offs_n * stride_h2o_seq
        tl.store(h2o_ptrs, qk, mask=mask_n)
        
        # Online Softmax 迭代更新
        m_ij = tl.max(qk, axis=0)
        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        p = tl.exp(qk - m_next)
        
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha
        
        # 加载 V 块并加权累加
        v_ptrs = V_ptr + offset_v + offs_n[:, None] * stride_v_seq + offs_d[None, :] * stride_v_dim
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        acc += tl.sum(p[:, None] * v, axis=0)
        
        m_i = m_next
        
    # 5. 写回 Output
    acc = acc / l_i
    tl.store(Out_ptr + offset_o + offs_d * stride_o_dim, acc, mask=mask_d)
    
    # --- Pass 2: 仅对 Score 进行归一化 (仅 1D 读写) ---
    for start_n in range(0, kv_len, BLOCK_SIZE_KV):
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_KV)
        mask_n = offs_n < kv_len
        h2o_ptrs = H2O_Score_ptr + offset_h2o + offs_n * stride_h2o_seq
        raw_qk = tl.load(h2o_ptrs, mask=mask_n, other=-float('inf'))
        
        # 使用最终的 m_i 和 l_i 归一化
        final_probs = tl.exp(raw_qk - m_i) / l_i
        tl.store(h2o_ptrs, final_probs, mask=mask_n)

# ============================================================================
# Python Wrapper
# ============================================================================

def fused_h2o_decode_attention_v2(q, k, v, scale=None):
    batch_size, q_len, num_heads, head_dim = q.shape
    kv_len = k.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
        
    output = torch.empty((batch_size, 1, num_heads, head_dim), device=q.device, dtype=q.dtype)
    h2o_score = torch.empty((batch_size, num_heads, kv_len), device=q.device, dtype=q.dtype)
    
    # 自动对齐块大小
    BLOCK_SIZE_KV = 128
    BLOCK_SIZE_HEAD_DIM = triton.next_power_of_2(head_dim)
    
    grid = (batch_size, num_heads)
    
    fused_h2o_decode_attention_v2_kernel[grid](
        q, k, v, output, h2o_score,
        batch_size, num_heads, head_dim, kv_len,
        q.stride(0), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(2), output.stride(3),
        h2o_score.stride(0), h2o_score.stride(1), h2o_score.stride(2),
        scale,
        BLOCK_SIZE_KV=BLOCK_SIZE_KV,
        BLOCK_SIZE_HEAD_DIM=BLOCK_SIZE_HEAD_DIM,
        num_warps=8,
        num_stages=4
    )
    return output, h2o_score

# ============================================================================
# Baseline (PyTorch)
# ============================================================================

def baseline_h2o_decode_attention(q, k, v):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v)
    h2o_score = probs.squeeze(2) # [B, H, L]
    return output.transpose(1, 2).contiguous(), h2o_score

# ============================================================================
# Correctness & Benchmark Logic
# ============================================================================

def run_evaluation():
    device = "cuda"
    dtype = torch.float16
    kv_lens = [2048, 4096, 8192]
    B, H, D = 1, 32, 128
    
    print(f"{'Context':<10} | {'Impl':<10} | {'Time (ms)':<12} | {'TFLOPS':<10} | {'Speedup':<10}")
    print("-" * 65)

    for kv_len in kv_lens:
        q = torch.randn(B, 1, H, D, device=device, dtype=dtype)
        k = torch.randn(B, kv_len, H, D, device=device, dtype=dtype)
        v = torch.randn(B, kv_len, H, D, device=device, dtype=dtype)
        
        # 正确性检查
        out_ref, h2o_ref = baseline_h2o_decode_attention(q, k, v)
        out_fused, h2o_fused = fused_h2o_decode_attention_v2(q, k, v)
        
        assert torch.allclose(out_fused, out_ref, atol=1e-2, rtol=1e-2), f"Output mismatch at {kv_len}"
        assert torch.allclose(h2o_fused, h2o_ref, atol=1e-2, rtol=1e-2), f"H2O Score mismatch at {kv_len}"

        # 性能测试 (Baseline)
        for _ in range(20): baseline_h2o_decode_attention(q, k, v) # warmup
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100): baseline_h2o_decode_attention(q, k, v)
        torch.cuda.synchronize()
        base_time = (time.time() - t0) * 10 # ms (1000/100)
        
        # 性能测试 (Fused)
        for _ in range(20): fused_h2o_decode_attention_v2(q, k, v) # warmup
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(100): fused_h2o_decode_attention_v2(q, k, v)
        torch.cuda.synchronize()
        fused_time = (time.time() - t1) * 10
        
        # 计算 TFLOPS (2 * B * H * L * D * 2 / time)
        flops = 4 * B * H * kv_len * D
        base_tflops = flops / (base_time * 1e-3) / 1e12
        fused_tflops = flops / (fused_time * 1e-3) / 1e12
        
        print(f"{kv_len:<10} | {'Baseline':<10} | {base_time:<12.4f} | {base_tflops:<10.3f} | 1.00x")
        print(f"{'':<10} | {'Fused-V2':<10} | {fused_time:<12.4f} | {fused_tflops:<10.3f} | {base_time/fused_time:.2f}x")
        print("-" * 65)

if __name__ == "__main__":
    run_evaluation()e