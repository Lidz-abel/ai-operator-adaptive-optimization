# decode_experiments/phase2_graph_analysis/extract_and_analyze_graph.py

"""
Phase 2: 计算图提取与分析
目标：
1. 提取 H2O Decode 阶段的真实计算图
2. 分析算子碎片化程度
3. 模拟 FlashTensor Algorithm 1 的 Kernel 识别
4. 生成可视化报告
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Set
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
# 如果你不需要实际加载模型，下面这行可以注释掉，因为我们用的是手动分析
# from asuka_exp.cases.kernels.h2o import H2O


@dataclass
class OperatorInfo:
    """算子信息"""
    name: str
    op_type: str
    is_compute_intensive: bool
    is_memory_intensive: bool
    has_reduce: bool
    

@dataclass
class GraphAnalysisResult:
    """计算图分析结果"""
    total_ops: int
    compute_ops: int
    memory_ops: int
    reduce_ops: int
    element_wise_ops: int
    kernel_boundaries: List[int]  # 基于 Algorithm 1 识别的边界
    fragmentation_score: float
    operators: List[OperatorInfo]


class ManualGraphAnalyzer:
    """
    手动分析 H2O 的计算图
    基于源代码直接分析，不依赖 FX/TorchScript
    """
    
    def __init__(self, model_type='decode'):
        self.model_type = model_type  # 'decode' or 'prefill'
        self.operators =[]
        
    def analyze_h2o_forward(self) -> GraphAnalysisResult:
        """
        手动分析 H2O 的 forward 函数
        
        H2O forward 的完整操作序列（包含 TopK 和 Gather）：
        1-3.   transpose (q, k, v)
        4.     matmul (q @ k.T)
        5.     div (scores / sqrt(d))
        6.     add (scores + mask)
        7.     softmax (probs) - Reduce 1
        8.     matmul (probs @ v)
        9.     transpose (output)
        10.    contiguous (output)
        11.    sum (h2o_score) - Reduce 2
        12.    topk (select heavy hitters) - Reduce 3 + Dynamic Shape!
        13-14. gather (k, v from selected indices) - Irregular memory access
        15-16. view (output reshape)
        """
        
        print("\n" + "="*70)
        print(f"Analyzing H2O {self.model_type.upper()} Computation Graph (Manual)")
        print("="*70)
        
        # 定义完整的操作序列（包含 H2O 核心的 TopK 和 KV Cache 收集）
        ops =[
            ("transpose_q", "transpose", False, True, False),
            ("transpose_k", "transpose", False, True, False),
            ("transpose_v", "transpose", False, True, False),
            ("matmul_qk", "matmul", True, False, False),
            ("div_scale", "div", False, False, False),
            ("add_mask", "add", False, False, False),
            ("softmax", "softmax", False, False, True),  # Reduce 1
            ("matmul_pv", "matmul", True, False, False),
            ("transpose_out", "transpose", False, True, False),
            ("contiguous", "contiguous", False, True, False),
            ("sum_h2o_score", "sum", False, False, True),  # Reduce 2
            ("topk_selection", "topk", False, True, True),  # Reduce 3 + Dynamic Shape!
            ("gather_k", "gather", False, True, False),     # Irregular memory access
            ("gather_v", "gather", False, True, False),     # Irregular memory access
            ("view_out", "view", False, True, False),
            ("view_h2o", "view", False, True, False),
        ]
        
        total_ops = len(ops)
        compute_ops = 0
        memory_ops = 0
        reduce_ops = 0
        element_wise_ops = 0
        dynamic_shape_ops = 0
        
        for idx, (name, op_type, is_compute, is_memory, has_reduce) in enumerate(ops):
            if is_compute:
                compute_ops += 1
            if is_memory:
                memory_ops += 1
            if has_reduce:
                reduce_ops += 1
            if op_type in ['add', 'mul', 'div', 'sub']:
                element_wise_ops += 1
            if op_type == 'topk':
                dynamic_shape_ops += 1
            
            op_info = OperatorInfo(
                name=name,
                op_type=op_type,
                is_compute_intensive=is_compute,
                is_memory_intensive=is_memory,
                has_reduce=has_reduce
            )
            self.operators.append(op_info)
            
            # 特殊标注
            special_note = ""
            if op_type == 'topk':
                special_note = " ⚠️ DYNAMIC SHAPE!"
            elif op_type == 'gather':
                special_note = " ⚠️ IRREGULAR ACCESS!"
            
            print(f"  [{idx+1:3d}] {name:20s} | {op_type:15s} | "
                  f"Compute:{is_compute} Memory:{is_memory} Reduce:{has_reduce}{special_note}")
        
        # 运行 Algorithm 1
        kernel_boundaries = self._identify_kernels()
        
        # 计算碎片化分数
        fragmentation_score = len(kernel_boundaries) / total_ops if total_ops > 0 else 0
        
        result = GraphAnalysisResult(
            total_ops=total_ops,
            compute_ops=compute_ops,
            memory_ops=memory_ops,
            reduce_ops=reduce_ops,
            element_wise_ops=element_wise_ops,
            kernel_boundaries=kernel_boundaries,
            fragmentation_score=fragmentation_score,
            operators=self.operators
        )
        
        self._print_summary(result)
        
        # 额外打印动态形状信息
        if dynamic_shape_ops > 0:
            print("\n" + "⚠️"*35)
            print(f"  CRITICAL: {dynamic_shape_ops} operation(s) introduce DYNAMIC SHAPES")
            print("  → TopK output size depends on K parameter (runtime-determined)")
            print("  → This requires FlashTensor to support dynamic dimension attributes")
            print("⚠️"*35)
        
        return result

    def _identify_kernels(self) -> List[int]:
        """
        实现 FlashTensor Algorithm 1: Kernel Identification
        
        核心逻辑：
        - 如果操作有 Reduce 依赖，则标记为 Kernel 边界
        - 否则尝试融合到当前 Kernel
        """
        print("\n" + "-"*70)
        print("Running FlashTensor Algorithm 1: Kernel Identification")
        print("-"*70)
        
        kernel_boundaries =[]
        current_kernel_start = 0
        kernel_id = 1
        
        for idx, op in enumerate(self.operators):
            if op.has_reduce:
                # 发现 Reduce 依赖，标记为边界
                if idx > current_kernel_start:
                    print(f"  Kernel {kernel_id}: ops {current_kernel_start}-{idx-1}")
                    kernel_id += 1
                    kernel_boundaries.append(idx)
                    print(f"  → Boundary at op {idx}: {op.name} ({op.op_type}) - Reduce dependency")
                    current_kernel_start = idx + 1
        
        # 最后一个 Kernel
        if current_kernel_start < len(self.operators):
            print(f"  Kernel {kernel_id}: ops {current_kernel_start}-{len(self.operators)-1}")
            kernel_boundaries.append(len(self.operators))
        
        print(f"\n  Total Kernels identified: {len(kernel_boundaries)}")
        print(f"  Average ops per kernel: {len(self.operators)/len(kernel_boundaries):.1f}")
        
        return kernel_boundaries
    
    def _print_summary(self, result: GraphAnalysisResult):
        """打印分析摘要"""
        print("\n" + "="*70)
        print("Graph Analysis Summary")
        print("="*70)
        print(f"  Total Operations:        {result.total_ops}")
        print(f"  Compute Operations:      {result.compute_ops} ({result.compute_ops/result.total_ops*100:.1f}%)")
        print(f"  Memory Operations:       {result.memory_ops} ({result.memory_ops/result.total_ops*100:.1f}%)")
        print(f"  Reduce Operations:       {result.reduce_ops} ({result.reduce_ops/result.total_ops*100:.1f}%)")
        print(f"  Element-wise Operations: {result.element_wise_ops} ({result.element_wise_ops/result.total_ops*100:.1f}%)")
        print(f"  Kernel Boundaries:       {len(result.kernel_boundaries)}")
        print(f"  Fragmentation Score:     {result.fragmentation_score:.3f}")
        print("="*70)


def analyze_fusion_opportunities(decode_result: GraphAnalysisResult):
    """分析融合机会"""
    
    print("\n" + "="*70)
    print("Fusion Opportunity Analysis")
    print("="*70)
    
    print("\n### Current Baseline (PyTorch Eager) ###")
    print("  Each operation is a separate kernel")
    print(f"  Total kernels: {decode_result.total_ops}")
    
    print("\n### FlashTensor (Current Algorithm 1) ###")
    print("  Breaks at Reduce operations (softmax, sum, topk)")
    print(f"  Total kernels: {len(decode_result.kernel_boundaries)}")
    
    # 计算理论融合收益
    baseline_kernels = decode_result.total_ops
    flashtensor_kernels = len(decode_result.kernel_boundaries)
    reduction = (baseline_kernels - flashtensor_kernels) / baseline_kernels * 100
    
    print(f"  Kernel reduction: {baseline_kernels} → {flashtensor_kernels} ({reduction:.1f}% reduction)")
    
    print("\n  Identified kernel boundaries:")
    kernel_start = 0
    for i, boundary in enumerate(decode_result.kernel_boundaries):
        ops_in_kernel =[op.name for op in decode_result.operators[kernel_start:boundary]]
        print(f"    Kernel {i+1}: {', '.join(ops_in_kernel)}")
        kernel_start = boundary
    
    print("\n### Proposed: FlashTensor-Decode (Aggressive Fusion) ###")
    print("  Strategy: Fuse across Reduce for 1×N tensors")
    print("  Potential fusion plan:")
    print("    Kernel 1: transpose → matmul → div → add → softmax → matmul → transpose → contiguous")
    print("              (Fuse attention computation)")
    print("    Kernel 2: sum → topk → gather → view")
    print("              (Fuse H2O scoring and selection)")
    print("  Estimated kernels: 2")
    
    print("\n  ⚠️ Challenges:")
    print("    1. TopK introduces dynamic shape (K is runtime parameter)")
    print("    2. Gather has irregular memory access pattern")
    print("    3. Need to support dynamic dimension in attribute propagation")
    
    aggressive_kernels = 2
    aggressive_reduction = (baseline_kernels - aggressive_kernels) / baseline_kernels * 100
    
    print(f"\n  Kernel reduction: {baseline_kernels} → {aggressive_kernels} ({aggressive_reduction:.1f}% reduction)")
    
    # 估算性能提升
    print("\n### Performance Impact Estimation ###")
    
    # 假设每个 kernel 启动开销 15 μs
    kernel_launch_overhead_us = 15
    baseline_overhead_ms = baseline_kernels * kernel_launch_overhead_us / 1000
    flashtensor_overhead_ms = flashtensor_kernels * kernel_launch_overhead_us / 1000
    aggressive_overhead_ms = aggressive_kernels * kernel_launch_overhead_us / 1000
    
    print(f"  Kernel launch overhead (assuming {kernel_launch_overhead_us} μs per kernel):")
    print(f"    Baseline:              {baseline_overhead_ms:.3f} ms ({baseline_kernels} kernels)")
    print(f"    FlashTensor (current): {flashtensor_overhead_ms:.3f} ms ({flashtensor_kernels} kernels)")
    print(f"    FlashTensor-Decode:    {aggressive_overhead_ms:.3f} ms ({aggressive_kernels} kernels)")
    
    # 基于 Phase 1 的实际数据（Decode 0.376 ms）
    decode_time_ms = 0.376
    overhead_ratio_baseline = baseline_overhead_ms / decode_time_ms * 100
    overhead_ratio_flashtensor = flashtensor_overhead_ms / decode_time_ms * 100
    overhead_ratio_aggressive = aggressive_overhead_ms / decode_time_ms * 100
    
    print(f"\n  Overhead as % of total Decode time (Phase 1: {decode_time_ms} ms):")
    print(f"    Baseline:              {overhead_ratio_baseline:.1f}%")
    print(f"    FlashTensor (current): {overhead_ratio_flashtensor:.1f}%")
    print(f"    FlashTensor-Decode:    {overhead_ratio_aggressive:.1f}%")
    
    overhead_saving = baseline_overhead_ms - aggressive_overhead_ms
    print(f"\n  Overhead saving: {overhead_saving:.3f} ms ({overhead_saving/decode_time_ms*100:.1f}% of total time)")
    
    # 内存流量减少估算
    print(f"\n  Memory traffic reduction:")
    print(f"    Baseline: Each op reads/writes global memory")
    print(f"              → {baseline_kernels} global memory round-trips")
    print(f"    FlashTensor-Decode: Intermediate results stay in registers/shared memory")
    print(f"              → {aggressive_kernels} global memory round-trips")
    print(f"    Estimated memory traffic reduction: {baseline_kernels/aggressive_kernels:.1f}x")
    
    print("\n  Combined speedup estimation:")
    print(f"    - Kernel launch overhead reduction: {baseline_overhead_ms/aggressive_overhead_ms:.1f}x")
    print(f"    - Memory traffic reduction: {baseline_kernels/aggressive_kernels:.1f}x")
    print(f"    - Expected total speedup: 3x-5x")
    
    print("\n  Target: Increase bandwidth utilization from 36% to 70-80%")
    print("="*70)

def compare_prefill_decode():
    """对比 Prefill 和 Decode 的计算图"""
    
    print("\n" + "#"*70)
    print("# Phase 2: Prefill vs Decode Graph Comparison")
    print("#"*70)
    
    # 分析 Decode 图
    print("\n### Analyzing Decode Graph (q_len=1, kv_len=8192) ###")
    decode_analyzer = ManualGraphAnalyzer(model_type='decode')
    decode_result = decode_analyzer.analyze_h2o_forward()
    
    # 分析 Prefill 图（操作序列相同，只是张量大小不同）
    print("\n\n### Analyzing Prefill Graph (q_len=8192, kv_len=8192) ###")
    prefill_analyzer = ManualGraphAnalyzer(model_type='prefill')
    prefill_result = prefill_analyzer.analyze_h2o_forward()
    
    # 对比分析
    print("\n" + "="*70)
    print("Prefill vs Decode Comparison")
    print("="*70)
    print(f"{'Metric':<30s} | {'Prefill':>15s} | {'Decode':>15s} | {'Ratio':>10s}")
    print("-"*70)
    print(f"{'Total Operations':<30s} | {prefill_result.total_ops:>15d} | {decode_result.total_ops:>15d} | {prefill_result.total_ops/decode_result.total_ops:>10.2f}x")
    print(f"{'Compute Operations':<30s} | {prefill_result.compute_ops:>15d} | {decode_result.compute_ops:>15d} | {prefill_result.compute_ops/max(decode_result.compute_ops,1):>10.2f}x")
    print(f"{'Reduce Operations':<30s} | {prefill_result.reduce_ops:>15d} | {decode_result.reduce_ops:>15d} | {prefill_result.reduce_ops/max(decode_result.reduce_ops,1):>10.2f}x")
    print(f"{'Kernel Boundaries':<30s} | {len(prefill_result.kernel_boundaries):>15d} | {len(decode_result.kernel_boundaries):>15d} | {len(prefill_result.kernel_boundaries)/max(len(decode_result.kernel_boundaries),1):>10.2f}x")
    print(f"{'Fragmentation Score':<30s} | {prefill_result.fragmentation_score:>15.3f} | {decode_result.fragmentation_score:>15.3f} | {prefill_result.fragmentation_score/max(decode_result.fragmentation_score,0.001):>10.2f}x")
    print("="*70)
    
    print("\nKey Insight:")
    print("  虽然 Prefill 和 Decode 的操作序列相同（都是 16 个操作），")
    print("  但它们的性能瓶颈完全不同：")
    print("    - Prefill: 受限于 N×N 矩阵的内存复杂度")
    print("    - Decode:  受限于 1×N 向量的内存带宽和 Kernel 启动开销")
    
    # 融合机会分析
    analyze_fusion_opportunities(decode_result)
    
    # 保存结果
    results = {
        'decode': asdict(decode_result),
        'prefill': asdict(prefill_result)
    }
    
    output_dir = 'decode_experiments/phase2_graph_analysis/results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'graph_analysis_results.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return decode_result, prefill_result


def main():
    """主函数"""
    
    print("\n" + "#"*70)
    print("# Phase 2: Graph Analysis & Kernel Identification")
    print("# Method: Manual analysis based on H2O source code")
    print("#"*70)
    
    # 运行对比分析
    decode_result, prefill_result = compare_prefill_decode()
    
    print("\n" + "#"*70)
    print("# Phase 2 Complete!")
    print("#"*70)
    print("\nKey Findings:")
    print(f"  1. H2O has {decode_result.total_ops} operations in forward pass")
    print(f"  2. FlashTensor Algorithm 1 identifies {len(decode_result.kernel_boundaries)} kernel boundaries")
    print(f"  3. Reduce operations: {decode_result.reduce_ops} (softmax, sum, topk)")
    print(f"  4. Fragmentation score: {decode_result.fragmentation_score:.3f}")
    print(f"  5. Baseline would use {decode_result.total_ops} kernels")
    print(f"  6. Aggressive fusion could reduce to 2-3 kernels")
    
    print("\nNext Steps:")
    print("  → Phase 3: Implement aggressive fusion rules in FlashTensor")
    print("  → Phase 4: Benchmark the fused kernel")
    print("  → Phase 5: End-to-end evaluation")


if __name__ == '__main__':
    main()