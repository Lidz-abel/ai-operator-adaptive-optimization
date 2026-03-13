#!/usr/bin/env python3
"""
Phase 1: 结果可视化脚本（增强版）
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
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
        self.system = self.data.get('system', 'torch')
        self.gpu_name = self.data.get('gpu_name', 'Unknown GPU')
        if 'theoretical_bandwidth_gbs' in self.data:
            self.theoretical_bw = float(self.data['theoretical_bandwidth_gbs'])
        elif self.prefill_results:
            self.theoretical_bw = float(self.prefill_results[0].get('theoretical_bandwidth_gbs', 0.0))
        else:
            self.theoretical_bw = 0.0

        prefill_ok = {r['context_len']: r for r in self.prefill_results if r.get('status', 'ok') == 'ok'}
        decode_ok = {r['context_len']: r for r in self.decode_results if r.get('status', 'ok') == 'ok'}
        contexts = sorted(set(prefill_ok.keys()) & set(decode_ok.keys()))
        self.valid_contexts = contexts
        self.prefill_ok = [prefill_ok[c] for c in contexts]
        self.decode_ok = [decode_ok[c] for c in contexts]
    
    def plot_all(self, output_dir: Path):
        """生成所有图表"""
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.valid_contexts:
            print("No valid context points to plot (all points may be OOM/skipped).")
            return
        
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

    @staticmethod
    def _pick(rec: Dict, primary: str, fallback: str) -> float:
        if primary in rec:
            return rec[primary]
        return rec[fallback]
    
    def plot_tflops_dive(self, output_dir: Path):
        """核心发现：TFLOPS "跳水"现象"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        context_lens = [r['context_len'] for r in self.prefill_ok]
        prefill_tflops = [self._pick(r, 'gpu_tflops_per_sec', 'gpu_tflops_per_sec') for r in self.prefill_ok]
        decode_tflops = [self._pick(r, 'gpu_tflops_per_sec', 'gpu_tflops_per_sec') for r in self.decode_ok]
        
        # 绘制曲线
        line1 = ax.plot(context_lens, prefill_tflops, 'o-', linewidth=2.5, 
                       markersize=10, color='#2E86AB', label='Prefill')
        line2 = ax.plot(context_lens, decode_tflops, 's-', linewidth=2.5, 
                       markersize=10, color='#A23B72', label='Decode')
        
        # 动态标注首尾变化，避免写死结论
        if len(context_lens) >= 2:
            first_pf = prefill_tflops[0]
            last_pf = prefill_tflops[-1]
            if last_pf > 0:
                drop_ratio = first_pf / last_pf
                ax.annotate(
                    f'Prefill first->last: {drop_ratio:.2f}x',
                    xy=(context_lens[-1], last_pf),
                    xytext=(context_lens[-1], last_pf * 1.25),
                    arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.8),
                    fontsize=10,
                    color='#2E86AB',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
                )
        
        ax.set_xlabel('Context Length', fontsize=13, fontweight='bold')
        ax.set_ylabel('Throughput (TFLOPS/s)', fontsize=13, fontweight='bold')
        ax.set_title(f'{self.model_name.upper()} - Computational Throughput Analysis\n'
                    f'System={self.system}, GPU={self.gpu_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(context_lens)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加数值标签
        y_base = max(max(prefill_tflops), max(decode_tflops), 1e-6)
        pf_off = y_base * 0.03
        dc_off = y_base * 0.015
        for i, (ctx, pf, dc) in enumerate(zip(context_lens, prefill_tflops, decode_tflops)):
            ax.text(ctx, pf + pf_off, f'{pf:.2f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color='#2E86AB')
            ax.text(ctx, dc + dc_off, f'{dc:.2f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color='#A23B72')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.model_name}_tflops_dive.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated: {self.model_name}_tflops_dive.png")
    
    def plot_bandwidth_trend(self, output_dir: Path):
        """内存带宽利用率趋势"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        context_lens = [r['context_len'] for r in self.prefill_ok]
        prefill_bw = [r['bandwidth_utilization_percent'] for r in self.prefill_ok]
        decode_bw = [r['bandwidth_utilization_percent'] for r in self.decode_ok]
        
        # 绘制曲线
        ax.plot(context_lens, prefill_bw, 'o-', linewidth=2.5, markersize=10, 
               color='#06A77D', label='Prefill')
        ax.plot(context_lens, decode_bw, 's-', linewidth=2.5, markersize=10, 
               color='#D5A021', label='Decode')
        
        ax.set_xlabel('Context Length', fontsize=13, fontweight='bold')
        ax.set_ylabel('Memory Bandwidth Utilization (%)', fontsize=13, fontweight='bold')
        if self.theoretical_bw > 0:
            subtitle = f'Theoretical BW: {self.theoretical_bw:.1f} GB/s'
        else:
            subtitle = 'Theoretical BW: unknown'
        ax.set_title(
            f'{self.model_name.upper()} - Memory Bandwidth Analysis\n{subtitle}',
            fontsize=14,
            fontweight='bold',
        )
        ax.set_xticks(context_lens)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加数值标签
        y_off = max(max(prefill_bw), max(decode_bw), 1.0) * 0.02
        for i, (ctx, pf, dc) in enumerate(zip(context_lens, prefill_bw, decode_bw)):
            ax.text(ctx, pf + y_off, f'{pf:.1f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color='#06A77D')
            ax.text(ctx, dc + y_off, f'{dc:.1f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color='#D5A021')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.model_name}_bandwidth_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated: {self.model_name}_bandwidth_trend.png")
    
    def plot_execution_time(self, output_dir: Path):
        """执行时间对比（GPU + CPU）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        context_lens = [r['context_len'] for r in self.prefill_ok]
        
        # GPU 计时
        prefill_gpu = [self._pick(r, 'gpu_median_time_ms', 'gpu_avg_time_ms') for r in self.prefill_ok]
        decode_gpu = [self._pick(r, 'gpu_median_time_ms', 'gpu_avg_time_ms') for r in self.decode_ok]
        
        # CPU 计时
        prefill_cpu = [self._pick(r, 'cpu_median_time_ms', 'cpu_avg_time_ms') for r in self.prefill_ok]
        decode_cpu = [self._pick(r, 'cpu_median_time_ms', 'cpu_avg_time_ms') for r in self.decode_ok]
        
        x = np.arange(len(context_lens))
        width = 0.35
        
        # GPU 计时图
        bars1 = ax1.bar(x - width/2, prefill_gpu, width, label='Prefill', 
                       alpha=0.8, color='#2E86AB', edgecolor='black', linewidth=1.2)
        bars2 = ax1.bar(x + width/2, decode_gpu, width, label='Decode', 
                       alpha=0.8, color='#A23B72', edgecolor='black', linewidth=1.2)
        
        ax1.set_xlabel('Context Length', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('GPU Event Timing (Median)', fontsize=13, fontweight='bold')
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
        ax2.set_title('CPU time.time() (Median)', fontsize=13, fontweight='bold')
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
        
        context_lens = [r['context_len'] for r in self.prefill_ok]
        gpu_ratios = [self._pick(p, 'gpu_median_time_ms', 'gpu_avg_time_ms') / self._pick(d, 'gpu_median_time_ms', 'gpu_avg_time_ms') 
                     for p, d in zip(self.prefill_ok, self.decode_ok)]
        cpu_ratios = [self._pick(p, 'cpu_median_time_ms', 'cpu_avg_time_ms') / self._pick(d, 'cpu_median_time_ms', 'cpu_avg_time_ms') 
                     for p, d in zip(self.prefill_ok, self.decode_ok)]
        
        # 理论比率（dense attention FLOPs 口径：Prefill/Decode ≈ N）
        theoretical_ratios = [n for n in context_lens]
        
        # 绘制曲线
        ax.plot(context_lens, gpu_ratios, 'o-', linewidth=2.5, markersize=10, 
               color='#2E86AB', label='GPU Event Ratio')
        ax.plot(context_lens, cpu_ratios, 's-', linewidth=2.5, markersize=10, 
               color='#06A77D', label='CPU time.time() Ratio')
        ax.plot(context_lens, theoretical_ratios, '--', linewidth=2, 
               color='gray', label='Theoretical Dense (N)', alpha=0.6)
        
        ax.set_xlabel('Context Length', fontsize=13, fontweight='bold')
        ax.set_ylabel('Prefill / Decode Time Ratio', fontsize=13, fontweight='bold')
        ax.set_title(f'{self.model_name.upper()} - Performance Gap Growth\n'
                    f'Actual vs Theoretical Ratio', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(context_lens)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加数值标签
        ratio_off = max(max(gpu_ratios), max(cpu_ratios), 1.0) * 0.03
        for i, (ctx, gpu_r, cpu_r, theo_r) in enumerate(zip(context_lens, gpu_ratios, cpu_ratios, theoretical_ratios)):
            ax.text(
                ctx,
                gpu_r + ratio_off,
                f'{gpu_r:.0f}x',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                color='#2E86AB',
            )
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.model_name}_time_ratio_growth.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated: {self.model_name}_time_ratio_growth.png")
    
    def plot_memory_usage(self, output_dir: Path):
        """显存使用对比"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        context_lens = [r['context_len'] for r in self.prefill_ok]
        prefill_mem = [self._pick(r, 'peak_memory_delta_mb', 'peak_memory_mb') for r in self.prefill_ok]
        decode_mem = [self._pick(r, 'peak_memory_delta_mb', 'peak_memory_mb') for r in self.decode_ok]
        
        x = np.arange(len(context_lens))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, prefill_mem, width, label='Prefill', 
                      alpha=0.8, color='#F18F01', edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, decode_mem, width, label='Decode', 
                      alpha=0.8, color='#C73E1D', edgecolor='black', linewidth=1.2)
        
        using_delta = any('peak_memory_delta_mb' in r for r in self.prefill_ok)
        ylabel = 'Peak Memory Delta (MB)' if using_delta else 'Peak Memory (MB)'
        
        ax.set_xlabel('Context Length', fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(f'{self.model_name.upper()} - GPU Memory Usage', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(context_lens)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        mem_off = max(max(prefill_mem), max(decode_mem), 1.0) * 0.02
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + mem_off,
                    f'{height:.0f} MB', ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + mem_off,
                    f'{height:.0f} MB', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.model_name}_memory_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated: {self.model_name}_memory_usage.png")
    
    def plot_dashboard(self, output_dir: Path):
        """综合仪表盘（4合1）"""
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        context_lens = [r['context_len'] for r in self.prefill_ok]
        x = np.arange(len(context_lens))
        width = 0.35
        
        # 1. TFLOPS 对比
        ax1 = fig.add_subplot(gs[0, 0])
        prefill_tflops = [self._pick(r, 'gpu_tflops_per_sec', 'gpu_tflops_per_sec') for r in self.prefill_ok]
        decode_tflops = [self._pick(r, 'gpu_tflops_per_sec', 'gpu_tflops_per_sec') for r in self.decode_ok]
        ax1.plot(context_lens, prefill_tflops, 'o-', linewidth=2, markersize=8, label='Prefill', color='#2E86AB')
        ax1.plot(context_lens, decode_tflops, 's-', linewidth=2, markersize=8, label='Decode', color='#A23B72')
        ax1.set_xlabel('Context Length', fontweight='bold')
        ax1.set_ylabel('TFLOPS/s', fontweight='bold')
        ax1.set_title('Computational Throughput', fontweight='bold', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 内存带宽
        ax2 = fig.add_subplot(gs[0, 1])
        prefill_bw = [r['bandwidth_utilization_percent'] for r in self.prefill_ok]
        decode_bw = [r['bandwidth_utilization_percent'] for r in self.decode_ok]
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
        prefill_time = [self._pick(r, 'gpu_median_time_ms', 'gpu_avg_time_ms') for r in self.prefill_ok]
        decode_time = [self._pick(r, 'gpu_median_time_ms', 'gpu_avg_time_ms') for r in self.decode_ok]
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
        prefill_mem = [self._pick(r, 'peak_memory_delta_mb', 'peak_memory_mb') for r in self.prefill_ok]
        decode_mem = [self._pick(r, 'peak_memory_delta_mb', 'peak_memory_mb') for r in self.decode_ok]
        ax4.bar(x - width/2, prefill_mem, width, label='Prefill', alpha=0.8, color='#F18F01')
        ax4.bar(x + width/2, decode_mem, width, label='Decode', alpha=0.8, color='#C73E1D')
        using_delta = any('peak_memory_delta_mb' in r for r in self.prefill_ok)
        ax4.set_xlabel('Context Length', fontweight='bold')
        ax4.set_ylabel('Peak Memory Delta (MB)' if using_delta else 'Peak Memory (MB)', fontweight='bold')
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
    print(f"  1. {visualizer.model_name}_tflops_dive.png - TFLOPS trend")
    print(f"  2. {visualizer.model_name}_bandwidth_trend.png - Bandwidth utilization trend")
    print(f"  3. {visualizer.model_name}_execution_time.png - Timing comparison")
    print(f"  4. {visualizer.model_name}_time_ratio_growth.png - Performance gap growth")
    print(f"  5. {visualizer.model_name}_memory_usage.png - GPU memory usage")
    print(f"  6. {visualizer.model_name}_dashboard.png - Comprehensive dashboard")


if __name__ == '__main__':
    main()
