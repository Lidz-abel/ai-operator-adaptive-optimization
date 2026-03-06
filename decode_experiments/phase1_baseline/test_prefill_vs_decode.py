#!/usr/bin/env python3
"""
Phase 1: Prefill vs Decode 基线性能对比测试（修正版）

关键修正：
- 分别测量 GPU Event 和 CPU time.time()，避免计时干扰
- CPU 计时完全遵循源代码的实现方式（torch.cuda.synchronize() 包围）
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from asuka_exp.cases.kernels import KERNEL_ZOO


class MemoryBandwidthProfiler:
    """内存带宽利用率分析器"""
    
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.theoretical_bandwidth_gbs = 936.0  # RTX 3090
    
    def estimate_memory_traffic(
        self, 
        batch_size: int,
        q_len: int, 
        kv_len: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype
    ) -> Dict[str, float]:
        """估算内存流量（GB）"""
        bytes_per_element = 2 if dtype == torch.float16 else 4
        
        q_read_bytes = batch_size * q_len * head_num * head_dim * bytes_per_element
        k_read_bytes = batch_size * kv_len * head_num * head_dim * bytes_per_element
        v_read_bytes = batch_size * kv_len * head_num * head_dim * bytes_per_element
        qk_rw_bytes = 2 * batch_size * head_num * q_len * kv_len * bytes_per_element
        softmax_rw_bytes = 2 * batch_size * head_num * q_len * kv_len * bytes_per_element
        output_write_bytes = batch_size * q_len * head_num * head_dim * bytes_per_element
        
        total_bytes = (
            q_read_bytes + k_read_bytes + v_read_bytes +
            qk_rw_bytes + softmax_rw_bytes + output_write_bytes
        )
        
        return {
            'q_read_gb': q_read_bytes / (1024 ** 3),
            'k_read_gb': k_read_bytes / (1024 ** 3),
            'v_read_gb': v_read_bytes / (1024 ** 3),
            'qk_rw_gb': qk_rw_bytes / (1024 ** 3),
            'softmax_rw_gb': softmax_rw_bytes / (1024 ** 3),
            'output_write_gb': output_write_bytes / (1024 ** 3),
            'total_gb': total_bytes / (1024 ** 3)
        }
    
    def calculate_bandwidth_utilization(
        self,
        memory_traffic_gb: float,
        time_ms: float
    ) -> Dict[str, float]:
        """计算带宽利用率"""
        time_s = time_ms / 1000.0
        achieved_bandwidth_gbs = memory_traffic_gb / time_s
        utilization_percent = (achieved_bandwidth_gbs / self.theoretical_bandwidth_gbs) * 100
        
        return {
            'achieved_bandwidth_gbs': achieved_bandwidth_gbs,
            'theoretical_bandwidth_gbs': self.theoretical_bandwidth_gbs,
            'utilization_percent': utilization_percent
        }


class FLOPsCalculator:
    """FLOPs 计算器"""
    
    @staticmethod
    def calculate_attention_flops(
        batch_size: int,
        q_len: int,
        kv_len: int,
        head_num: int,
        head_dim: int,
        is_prefill: bool = False
    ) -> Dict[str, float]:
        """计算 Attention 的 FLOPs"""
        base_flops = 4 * batch_size * head_num * q_len * kv_len * head_dim
        
        if is_prefill:
            effective_flops = base_flops / 2
            reason = "Prefill with causal mask (aligned with source code)"
        else:
            effective_flops = base_flops
            reason = "Decode without causal mask effect"
        
        return {
            'base_flops': base_flops,
            'effective_flops': effective_flops,
            'base_gflops': base_flops / 1e9,
            'effective_gflops': effective_flops / 1e9,
            'reason': reason
        }


class PrefillDecodeComparison:
    """Prefill vs Decode 性能对比测试"""
    
    def __init__(
        self,
        model_name: str = 'h2o',
        batch_size: int = 1,
        head_num: int = 32,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
        device: str = 'cuda:0'
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        assert model_name in KERNEL_ZOO, f"Model {model_name} not found"
        model_cls = KERNEL_ZOO[model_name]
        self.model = model_cls(
            kv_head_num=head_num,
            head_num=head_num,
            head_dim=head_dim
        ).eval().to(device)
        
        self.bandwidth_profiler = MemoryBandwidthProfiler()
        self.flops_calculator = FLOPsCalculator()
    
    def run_single_test(
        self,
        q_len: int,
        kv_len: int,
        is_prefill: bool,
        warmup: int = 10,
        runs: int = 50
    ) -> Dict:
        """运行单次测试（修正版：分别测量 GPU 和 CPU 时间）"""
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 准备输入数据
        q = torch.randn(
            self.batch_size, q_len, self.head_num, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            self.batch_size, kv_len, self.head_num, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            self.batch_size, kv_len, self.head_num, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        
        # Warmup
        print(f"  Warmup ({warmup} runs)...", flush=True)
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(q, k, v)
        torch.cuda.synchronize()
        
        # ===== GPU Event 计时 =====
        print(f"  Measuring with GPU Event ({runs} runs)...", flush=True)
        gpu_times_ms = []
        
        for _ in range(runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            with torch.no_grad():
                _ = self.model(q, k, v)
            end_event.record()
            torch.cuda.synchronize()
            
            gpu_times_ms.append(start_event.elapsed_time(end_event))
        
        # ===== CPU time.time() 计时（与源代码对齐）=====
        print(f"  Measuring with CPU time.time() ({runs} runs)...", flush=True)
        cpu_times_ms = []
        
        for _ in range(runs):
            torch.cuda.synchronize()
            tik = time.time()
            with torch.no_grad():
                _ = self.model(q, k, v)
            torch.cuda.synchronize()
            tok = time.time()
            cpu_times_ms.append((tok - tik) * 1000.0)
        
        # 统计
        gpu_times_ms = np.array(gpu_times_ms)
        gpu_avg = np.mean(gpu_times_ms)
        gpu_min = np.min(gpu_times_ms)
        gpu_max = np.max(gpu_times_ms)
        gpu_std = np.std(gpu_times_ms)
        
        cpu_times_ms = np.array(cpu_times_ms)
        cpu_avg = np.mean(cpu_times_ms)
        cpu_min = np.min(cpu_times_ms)
        cpu_max = np.max(cpu_times_ms)
        cpu_std = np.std(cpu_times_ms)
        
        # 计算 FLOPs
        flops_info = self.flops_calculator.calculate_attention_flops(
            self.batch_size, q_len, kv_len, self.head_num, self.head_dim, is_prefill
        )
        
        gpu_tflops_per_sec = flops_info['effective_gflops'] / (gpu_avg / 1000.0) / 1000.0
        cpu_tflops_per_sec = flops_info['effective_gflops'] / (cpu_avg / 1000.0) / 1000.0
        
        # 内存带宽
        memory_breakdown = self.bandwidth_profiler.estimate_memory_traffic(
            self.batch_size, q_len, kv_len, self.head_num, self.head_dim, self.dtype
        )
        bandwidth_stats = self.bandwidth_profiler.calculate_bandwidth_utilization(
            memory_breakdown['total_gb'], gpu_avg
        )
        
        # 显存
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        with torch.no_grad():
            _ = self.model(q, k, v)
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        return {
            'q_len': q_len,
            'kv_len': kv_len,
            'is_prefill': is_prefill,
            'gpu_avg_time_ms': float(gpu_avg),
            'gpu_min_time_ms': float(gpu_min),
            'gpu_max_time_ms': float(gpu_max),
            'gpu_std_time_ms': float(gpu_std),
            'gpu_tflops_per_sec': float(gpu_tflops_per_sec),
            'cpu_avg_time_ms': float(cpu_avg),
            'cpu_min_time_ms': float(cpu_min),
            'cpu_max_time_ms': float(cpu_max),
            'cpu_std_time_ms': float(cpu_std),
            'cpu_tflops_per_sec': float(cpu_tflops_per_sec),
            'base_gflops': float(flops_info['base_gflops']),
            'effective_gflops': float(flops_info['effective_gflops']),
            'flops_reason': flops_info['reason'],
            'memory_traffic_gb': float(memory_breakdown['total_gb']),
            'memory_breakdown': {k: float(v) for k, v in memory_breakdown.items()},
            'achieved_bandwidth_gbs': float(bandwidth_stats['achieved_bandwidth_gbs']),
            'theoretical_bandwidth_gbs': float(bandwidth_stats['theoretical_bandwidth_gbs']),
            'bandwidth_utilization_percent': float(bandwidth_stats['utilization_percent']),
            'peak_memory_mb': float(peak_memory_mb)
        }
    
    def run_prefill_test(self, context_len: int, **kwargs) -> Dict:
        print(f"\n[Prefill] context_len={context_len}")
        result = self.run_single_test(q_len=context_len, kv_len=context_len, is_prefill=True, **kwargs)
        result['stage'] = 'prefill'
        result['context_len'] = context_len
        return result
    
    def run_decode_test(self, context_len: int, **kwargs) -> Dict:
        print(f"\n[Decode] context_len={context_len}")
        result = self.run_single_test(q_len=1, kv_len=context_len, is_prefill=False, **kwargs)
        result['stage'] = 'decode'
        result['context_len'] = context_len
        return result
    
    def compare(self, context_lengths: List[int], **kwargs) -> Dict:
        results = {
            'model': self.model_name,
            'batch_size': self.batch_size,
            'head_num': self.head_num,
            'head_dim': self.head_dim,
            'dtype': str(self.dtype),
            'device': self.device,
            'gpu_name': torch.cuda.get_device_name(),
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
            'prefill_results': [],
            'decode_results': []
        }
        
        for context_len in context_lengths:
            print(f"\n{'='*70}")
            print(f"Testing context_len = {context_len}")
            print(f"{'='*70}")
            
            prefill_result = self.run_prefill_test(context_len, **kwargs)
            results['prefill_results'].append(prefill_result)
            
            decode_result = self.run_decode_test(context_len, **kwargs)
            results['decode_results'].append(decode_result)
            
            self._print_comparison(prefill_result, decode_result)
        
        return results
    
    def _print_comparison(self, prefill: Dict, decode: Dict):
        print(f"\n{'─'*70}")
        print(f"Comparison Summary (context_len={prefill['context_len']})")
        print(f"{'─'*70}")
        
        print(f"\n{'GPU Event Timing (Precise)'}")
        print(f"{'─'*70}")
        print(f"{'Stage':<15} {'Time (ms)':<15} {'TFLOPS/s':<15} {'BW Util %':<15}")
        print(f"{'-'*70}")
        print(f"{'Prefill':<15} {prefill['gpu_avg_time_ms']:<15.4f} "
              f"{prefill['gpu_tflops_per_sec']:<15.4f} "
              f"{prefill['bandwidth_utilization_percent']:<15.2f}")
        print(f"{'Decode':<15} {decode['gpu_avg_time_ms']:<15.4f} "
              f"{decode['gpu_tflops_per_sec']:<15.4f} "
              f"{decode['bandwidth_utilization_percent']:<15.2f}")
        
        print(f"\n{'CPU time.time() Timing (Source Code Aligned)'}")
        print(f"{'─'*70}")
        print(f"{'Stage':<15} {'Time (ms)':<15} {'TFLOPS/s':<15}")
        print(f"{'-'*70}")
        print(f"{'Prefill':<15} {prefill['cpu_avg_time_ms']:<15.4f} "
              f"{prefill['cpu_tflops_per_sec']:<15.4f}")
        print(f"{'Decode':<15} {decode['cpu_avg_time_ms']:<15.4f} "
              f"{decode['cpu_tflops_per_sec']:<15.4f}")
        
        gpu_time_ratio = prefill['gpu_avg_time_ms'] / decode['gpu_avg_time_ms']
        cpu_time_ratio = prefill['cpu_avg_time_ms'] / decode['cpu_avg_time_ms']
        
        print(f"\n{'Key Metrics'}")
        print(f"{'─'*70}")
        print(f"{'Metric':<40} {'Value':<20}")
        print(f"{'-'*70}")
        print(f"{'Prefill/Decode Time Ratio (GPU)':<40} {gpu_time_ratio:<20.2f}x")
        print(f"{'Prefill/Decode Time Ratio (CPU)':<40} {cpu_time_ratio:<20.2f}x")
        print(f"{'Prefill Effective GFLOPs':<40} {prefill['effective_gflops']:<20.2f}")
        print(f"{'Decode Effective GFLOPs':<40} {decode['effective_gflops']:<20.2f}")
        print(f"{'Prefill Memory Traffic (GB)':<40} {prefill['memory_traffic_gb']:<20.4f}")
        print(f"{'Decode Memory Traffic (GB)':<40} {decode['memory_traffic_gb']:<20.4f}")
        print(f"{'Prefill Peak Memory (MB)':<40} {prefill['peak_memory_mb']:<20.2f}")
        print(f"{'Decode Peak Memory (MB)':<40} {decode['peak_memory_mb']:<20.2f}")


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Prefill vs Decode Performance Comparison')
    parser.add_argument('--model', '-m', type=str, default='h2o', choices=['attn', 'h2o', 'roco', 'keyformer'])
    parser.add_argument('--context_lengths', '-c', type=int, nargs='+', default=[2048, 4096, 8192])
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--output', '-o', type=str, default=None)
    
    args = parser.parse_args()
    
    print("="*70)
    print("Phase 1: Prefill vs Decode Performance Comparison")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Context Lengths: {args.context_lengths}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Warmup Runs: {args.warmup}")
    print(f"Measurement Runs: {args.runs}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print("="*70)
    print("\nTiming Methods:")
    print("  1. GPU Event (torch.cuda.Event) - Precise GPU timing")
    print("  2. CPU time.time() - Aligned with source code")
    print("\nFLOPs Calculation:")
    print("  - Prefill: 4*B*H*N*N*D / 2 (causal mask, aligned with source)")
    print("  - Decode: 4*B*H*1*N*D (correct formula)")
    print("="*70)
    
    comparison = PrefillDecodeComparison(model_name=args.model, batch_size=args.batch_size)
    results = comparison.compare(context_lengths=args.context_lengths, warmup=args.warmup, runs=args.runs)
    
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / 'decode_experiments' / 'phase1_baseline' / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'prefill_vs_decode_{args.model}_{timestamp}.json'
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
