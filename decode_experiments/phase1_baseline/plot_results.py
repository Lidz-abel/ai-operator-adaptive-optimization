#!/usr/bin/env python3
"""
Phase 1: 结果可视化脚本（增强版）

重点突出：
1. N=8192 时 Prefill 的性能"崩塌"
2. Decode 的内存带宽瓶颈
3. TFLOPS 曲线的"跳水"现象
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 设置字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False


class EnhancedResultVisualizer:
    """增强版结果可视化器"""
    
    def __init__(self, result_file: Path):
        self.result_file = result_file
        with open(result_file, 'r') as f:
            self.data = json.load(f)
        
        self.model_name = self.data['model']
        self.prefill_results = self.data['prefill_results']
        self.decode_results = self.data['decode_results']
    
    def plot_all(self, output_dir: Path):
        """生成所有图表"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating enhanced plots for {self.model_name}...")
        
        # 1. 核心发现：TFLOPS "跳水"现象
        self.plot_tflops_dive(output_dir)
        
        # 2. 内存带宽利用率趋势
        self.plot_bandwidth_trend(output_dir)
        
        # 3. 执行时间对比（GPU + CPU）
        self.plot_execution_time(output_dir)
        
        # 4. Prefill/Decode 时间比率增长
        self.plot_time_ratio_growth(output_dir)
        
        # 5. 显存使用对比
        self.plot_memory_usage(output_dir)
        
        # 6. 综合仪表盘（4合1）
        self.plot_dashboard(output_dir)
        
        print(f"All plots saved to {output_dir}/")
    
    def plot_tflops_dive(self, output_dir: Path):
        """核心发现：TFLOPS "跳水"现象"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        context_lens = [r['context_len'] for r in self.prefill_results]
        prefill_tflops = [r['gpu_tflops_per_sec'] for r in self.prefill_results]
        decode_tflops = [r['gpu_tflops_per_sec'] for r in self.decode_results]
        
        # 绘制曲线
        line1 = ax.plot(context_lens, prefill_tflops, 'o-', linewidth=2.5, 
                       markersize=10, color='#2E86AB', label='Prefill')
        line2 = ax.plot(context_lens, decode_tflops, 's-', linewidth=2.5, 
                       markersize=10, color='#A23B72', label='Decode')
        
        # 标注 8192 的"跳水"点
        if len(context_lens) >= 3:
            ax.annotate('Performance Dive!\n(15x slowdown)', 
                       xy=(context_lens[2], prefill_tflops[2]),
                       xytext=(context_lens[2] - 1500, prefill_tflops[2] + 1.5),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=11, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        # 添加理论峰值参考线
        rtx3090_peak = 35.6  # RTX 3090 FP16 Tensor Core 峰值（实际可达）
        ax.axhline(y=rtx3090_peak, color='gray', linestyle='--', linewidth=1.5, 
                  label=f'RTX 3090 FP16 Peak (~{rtx3090_peak} TFLOPS)', alpha=0.6)
        
        ax.set_xlabel('Context Length', fontsize=13, fontweight='bold')
        ax.set_ylabel('Throughput (TFLOPS/s)', fontsize=13, fontweight='bold')
        ax.set_title(f'{self.model_name.upper()} - Computational Throughput Analysis\n'
                    f'Key Finding: Prefill Performance Collapse at N=8192', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(context_lens)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for i, (ctx, pf, dc) in enumerate(zip(context_lens, prefill_tflops, decode_tflops)):
            ax.text(ctx, pf + 0.2, f'{pf:.2f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color='#2E86AB')
            ax.text(ctx, dc + 0.02, f'{dc:.2f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color='#A23B72')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.model_name}_tflops_dive.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated: {self.model_name}_tflops_dive.png")
    
    def plot_bandwidth_trend(self, output_dir: Path):
        """内存带宽利用率趋势"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        context_lens = [r['context_len'] for r in self.prefill_results]
        prefill_bw = [r['bandwidth_utilization_percent'] for r in self.prefill_results]
        decode_bw = [r['bandwidth_utilization_percent'] for r in self.decode_results]
        
        # 绘制曲线
        ax.plot(context_lens, prefill_bw, 'o-', linewidth=2.5, markersize=10, 
               color='#06A77D', label='Prefill')
        ax.plot(context_lens, decode_bw, 's-', linewidth=2.5, markersize=10, 
               color='#D5A021', label='Decode')
        
        # 标注 Decode 的带宽瓶颈
        if len(context_lens) >= 3:
            ax.annotate('Memory Bandwidth\nBottleneck (36%)', 
                       xy=(context_lens[2], decode_bw[2]),
                       xytext=(context_lens[2] - 1500, decode_bw[2] - 8),
                       arrowprops=dict(arrowstyle='->', color='#D5A021', lw=2),
                       fontsize=11, color='#D5A021', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            
            # 标注 Prefill 的带宽崩塌
            ax.annotate('Compute-Bound\n(Only 4.4%)', 
                       xy=(context_lens[2], prefill_bw[2]),
                       xytext=(context_lens[2] + 500, prefill_bw[2] + 5),
                       arrowprops=dict(arrowstyle='->', color='#06A77D', lw=2),
                       fontsize=11, color='#06A77D', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        ax.set_xlabel('Context Length', fontsize=13, fontweight='bold')
        ax.set_ylabel('Memory Bandwidth Utilization (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'{self.model_name.upper()} - Memory Bandwidth Analysis\n'
                    f'RTX 3090 Theoretical: 936 GB/s', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(context_lens)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for i, (ctx, pf, dc) in enumerate(zip(context_lens, prefill_bw, decode_bw)):
            ax.text(ctx, pf + 1, f'{pf:.1f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color='#06A77D')
            ax.text(ctx, dc + 1, f'{dc:.1f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color='#D5A021')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.model_name}_bandwidth_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated: {self.model_name}_bandwidth_trend.png")
    
    def plot_execution_time(self, output_dir: Path):
        """执行时间对比（GPU + CPU）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        context_lens = [r['context_len'] for r in self.prefill_results]
        
        # GPU 计时
        prefill_gpu = [r['gpu_avg_time_ms'] for r in self.prefill_results]
        decode_gpu = [r['gpu_avg_time_ms'] for r in self.decode_results]
        
        # CPU 计时
        prefill_cpu = [r['cpu_avg_time_ms'] for r in self.prefill_results]
        decode_cpu = [r['cpu_avg_time_ms'] for r in self.decode_results]
        
        x = np.arange(len(context_lens))
        width = 0.35
        
        # GPU 计时图
        bars1 = ax1.bar(x - width/2, prefill_gpu, width, label='Prefill', 
                       alpha=0.8, color='#2E86AB', edgecolor='black', linewidth=1.2)
        bars2 = ax1.bar(x + width/2, decode_gpu, width, label='Decode', 
                       alpha=0.8, color='#A23B72', edgecolor='black', linewidth=1.2)
        
        ax1.set_xlabel('Context Length', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('GPU Event Timing (Precise)', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(context_lens)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # CPU 计时图
        bars3 = ax2.bar(x - width/2, prefill_cpu, width, label='Prefill', 
                       alpha=0.8, color='#06A77D', edgecolor='black', linewidth=1.2)
        bars4 = ax2.bar(x + width/2, decode_cpu, width, label='Decode', 
                       alpha=0.8, color='#D5A021', edgecolor='black', linewidth=1.2)
        
        ax2.set_xlabel('Context Length', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('CPU time.time() (Source Aligned)', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(context_lens)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        fig.suptitle(f'{self.model_name.upper()} - Execution Time Comparison', 
                    fontsize=15, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.model_name}_execution_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated: {self.model_name}_execution_time.png")
    
    def plot_time_ratio_growth(self, output_dir: Path):
        """Prefill/Decode 时间比率增长"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        context_lens = [r['context_len'] for r in self.prefill_results]
        gpu_ratios = [p['gpu_avg_time_ms'] / d['gpu_avg_time_ms'] 
                     for p, d in zip(self.prefill_results, self.decode_results)]
        cpu_ratios = [p['cpu_avg_time_ms'] / d['cpu_avg_time_ms'] 
                     for p, d in zip(self.prefill_results, self.decode_results)]
        
        # 理论比率（N/2）
        theoretical_ratios = [n / 2 for n in context_lens]
        
        # 绘制曲线
        ax.plot(context_lens, gpu_ratios, 'o-', linewidth=2.5, markersize=10, 
               color='#2E86AB', label='GPU Event Ratio')
        ax.plot(context_lens, cpu_ratios, 's-', linewidth=2.5, markersize=10, 
               color='#06A77D', label='CPU time.time() Ratio')
        ax.plot(context_lens, theoretical_ratios, '--', linewidth=2, 
               color='gray', label='Theoretical (N/2)', alpha=0.6)
        
        ax.set_xlabel('Context Length', fontsize=13, fontweight='bold')
        ax.set_ylabel('Prefill / Decode Time Ratio', fontsize=13, fontweight='bold')
        ax.set_title(f'{self.model_name.upper()} - Performance Gap Growth\n'
                    f'Actual vs Theoretical Ratio', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(context_lens)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for i, (ctx, gpu_r, cpu_r, theo_r) in enumerate(zip(context_lens, gpu_ratios, cpu_ratios, theoretical_ratios)):
            ax.text(ctx, gpu_r + 30, f'{gpu_r:.0f}x', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color='#2E86AB')
            ax.text(ctx, theo_r - 50, f'Theory: {theo_r:.0f}x', ha='center', va='top', 
                   fontsize=9, style='italic', color='gray')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.model_name}_time_ratio_growth.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated: {self.model_name}_time_ratio_growth.png")
    
    def plot_memory_usage(self, output_dir: Path):
        """显存使用对比"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        context_lens = [r['context_len'] for r in self.prefill_results]
        prefill_mem = [r['peak_memory_mb'] for r in self.prefill_results]
        decode_mem = [r['peak_memory_mb'] for r in self.decode_results]
        
        x = np.arange(len(context_lens))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, prefill_mem, width, label='Prefill', 
                      alpha=0.8, color='#F18F01', edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, decode_mem, width, label='Decode', 
                      alpha=0.8, color='#C73E1D', edgecolor='black', linewidth=1.2)
        
        # 添加 RTX 3090 显存上限参考线
        ax.axhline(y=24576, color='red', linestyle='--', linewidth=2, 
                  label='RTX 3090 Total (24 GB)', alpha=0.7)
        
        ax.set_xlabel('Context Length', fontsize=13, fontweight='bold')
        ax.set_ylabel('Peak Memory (MB)', fontsize=13, fontweight='bold')
        ax.set_title(f'{self.model_name.upper()} - GPU Memory Usage\n'
                    f'Prefill Approaching Memory Limit at N=8192', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(context_lens)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 500,
                    f'{height:.0f} MB', ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 500,
                    f'{height:.0f} MB', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.model_name}_memory_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated: {self.model_name}_memory_usage.png")
    
    def plot_dashboard(self, output_dir: Path):
        """综合仪表盘（4合1）"""
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        context_lens = [r['context_len'] for r in self.prefill_results]
        x = np.arange(len(context_lens))
        width = 0.35
        
        # 1. TFLOPS 对比
        ax1 = fig.add_subplot(gs[0, 0])
        prefill_tflops = [r['gpu_tflops_per_sec'] for r in self.prefill_results]
        decode_tflops = [r['gpu_tflops_per_sec'] for r in self.decode_results]
        ax1.plot(context_lens, prefill_tflops, 'o-', linewidth=2, markersize=8, label='Prefill', color='#2E86AB')
        ax1.plot(context_lens, decode_tflops, 's-', linewidth=2, markersize=8, label='Decode', color='#A23B72')
        ax1.set_xlabel('Context Length', fontweight='bold')
        ax1.set_ylabel('TFLOPS/s', fontweight='bold')
        ax1.set_title('Computational Throughput', fontweight='bold', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 内存带宽
        ax2 = fig.add_subplot(gs[0, 1])
        prefill_bw = [r['bandwidth_utilization_percent'] for r in self.prefill_results]
        decode_bw = [r['bandwidth_utilization_percent'] for r in self.decode_results]
        ax2.bar(x - width/2, prefill_bw, width, label='Prefill', alpha=0.8, color='#06A77D')
        ax2.bar(x + width/2, decode_bw, width, label='Decode', alpha=0.8, color='#D5A021')
        ax2.set_xlabel('Context Length', fontweight='bold')
        ax2.set_ylabel('Bandwidth Utilization (%)', fontweight='bold')
        ax2.set_title('Memory Bandwidth Utilization', fontweight='bold', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(context_lens)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 执行时间
        ax3 = fig.add_subplot(gs[1, 0])
        prefill_time = [r['gpu_avg_time_ms'] for r in self.prefill_results]
        decode_time = [r['gpu_avg_time_ms'] for r in self.decode_results]
        ax3.bar(x - width/2, prefill_time, width, label='Prefill', alpha=0.8, color='#2E86AB')
        ax3.bar(x + width/2, decode_time, width, label='Decode', alpha=0.8, color='#A23B72')
        ax3.set_xlabel('Context Length', fontweight='bold')
        ax3.set_ylabel('Time (ms)', fontweight='bold')
        ax3.set_title('GPU Execution Time', fontweight='bold', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(context_lens)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 显存使用
        ax4 = fig.add_subplot(gs[1, 1])
        prefill_mem = [r['peak_memory_mb'] for r in self.prefill_results]
        decode_mem = [r['peak_memory_mb'] for r in self.decode_results]
        ax4.bar(x - width/2, prefill_mem, width, label='Prefill', alpha=0.8, color='#F18F01')
        ax4.bar(x + width/2, decode_mem, width, label='Decode', alpha=0.8, color='#C73E1D')
        ax4.axhline(y=24576, color='red', linestyle='--', linewidth=1.5, label='GPU Limit', alpha=0.7)
        ax4.set_xlabel('Context Length', fontweight='bold')
        ax4.set_ylabel('Peak Memory (MB)', fontweight='bold')
        ax4.set_title('GPU Memory Usage', fontweight='bold', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(context_lens)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'{self.model_name.upper()} - Comprehensive Performance Dashboard', 
                    fontsize=16, fontweight='bold')
        
        plt.savefig(output_dir / f'{self.model_name}_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated: {self.model_name}_dashboard.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize Phase 1 results (Enhanced)')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input JSON result file')
    parser.add_argument('--output_dir', '-o', type=str, default='plots', help='Output directory')
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return
    
    output_dir = Path(args.output_dir)
    
    print("="*70)
    print("Phase 1: Enhanced Result Visualization")
    print("="*70)
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}/")
    print("="*70)
    
    visualizer = EnhancedResultVisualizer(input_file)
    visualizer.plot_all(output_dir)
    
    print("\n" + "="*70)
    print("✓ All enhanced plots generated successfully!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. {visualizer.model_name}_tflops_dive.png - TFLOPS performance collapse")
    print(f"  2. {visualizer.model_name}_bandwidth_trend.png - Memory bandwidth bottleneck")
    print(f"  3. {visualizer.model_name}_execution_time.png - Timing comparison")
    print(f"  4. {visualizer.model_name}_time_ratio_growth.png - Performance gap growth")
    print(f"  5. {visualizer.model_name}_memory_usage.png - GPU memory usage")
    print(f"  6. {visualizer.model_name}_dashboard.png - Comprehensive dashboard")


if __name__ == '__main__':
    main()
