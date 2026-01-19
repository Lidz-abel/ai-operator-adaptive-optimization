import sys
import os
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 确保能导入项目中的 plot_utils (如果原项目有这个依赖)
# 如果原项目 plot_utils 比较复杂，这里尽量使用标准库模拟关键颜色
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 定义模型和系统 (与 shell 脚本一致)
model_names = ['attn', 'h2o', 'gemma2']
sys_names = ['torch', 'our']

# 模拟原项目的配色 (根据原脚本逻辑简化)
# 0: Torch(Blueish), 1: Our(Redish/Orange)
COLOR_DEF = ['#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5', '#C49C94', '#F7B6D2']
HATCH_DEF = [None, '//', '\\\\', 'xx', '..', '++', '**']

# 显示名称映射
MODEL_DISPLAY_NAME = {
    'attn': 'Standard Attn',
    'h2o': 'H2O (Sparse)',
    'gemma2': 'Gemma2'
}
SYS_DISPLAY_NAME = {
    'torch': 'PyTorch',
    'our': 'FlashTensor'
}

def extract(log_dir):
    # Regex matching: e2e.rtx3090.attn.torch.log
    fn_pattern = r'(?P<exp>[^.]+)\.(?P<device>[^.]+)\.(?P<model>[^.]+)\.(?P<sys>[^.]+)\.log'
    # Regex matching time output
    # Kernel log usually has "gflops=..." or we look for execution time
    # E2E log usually has "avg: 1.2 ms"
    # 这里我们需要根据你的 run_kernel.py 和 run_e2e.py 的实际输出来调整正则
    # 假设 run_e2e.py 输出 "[perf] ... avg: 12.3 ms"
    # 假设 run_kernel.py 输出也包含类似的 avg 或者我们需要算 GFLOPS
    
    # 针对 E2E 的正则 (Latency)
    time_pattern = r'avg:\s+(?P<val>[\d.]+)' 
    
    # 针对 Kernel 的正则 (GFLOPS) - 根据之前的脚本输出 "gflops=xxxx"
    gflops_pattern = r'gflops=(?P<val>[\d.]+)'

    data = {
        "kernel": {"rtx3090": {}},
        "e2e": {"rtx3090": {}},
    }
    
    # Initialize
    for exp in data:
        for device in data[exp]:
            data[exp][device] = {sys: {model: 0.0 for model in model_names} for sys in sys_names}

    for root, _, files in os.walk(log_dir):
        for file in files:
            match = re.match(fn_pattern, file)
            if not match:
                continue
            
            exp = match.group('exp')
            device = match.group('device').lower()
            model = match.group('model')
            sys_key = match.group('sys')

            if device != 'rtx3090': continue
            if model not in model_names: continue
            if sys_key not in sys_names: continue

            fp = os.path.join(root, file)
            with open(fp, 'r') as f:
                content = f.read()
                
                val = 0.0
                if exp == 'e2e':
                    # E2E 找 Latency (越低越好)
                    m = re.search(time_pattern, content)
                    if m: val = float(m.group('val'))
                else:
                    # Kernel 找 GFLOPS (越高越好)
                    m = re.search(gflops_pattern, content)
                    if m: val = float(m.group('val'))
                
                data[exp][device][sys_key][model] = val

    return data

def plot_chart(data, exp_type, title, ylabel):
    device = 'rtx3090'
    df = pd.DataFrame(data[exp_type][device])
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    # Draw bars
    for i, sys_name in enumerate(sys_names):
        vals = [df[sys_name][m] for m in model_names]
        offset = width * i
        rects = ax.bar(x + offset, vals, width, label=SYS_DISPLAY_NAME[sys_name], 
                       color=COLOR_DEF[i], edgecolor='black', hatch=HATCH_DEF[i])
        
        # Add labels
        ax.bar_label(rects, padding=3, fmt='%.1f', fontsize=8)

    # Calculate Speedup (Our vs Torch)
    # 对于 Latency (E2E): Torch / Our
    # 对于 GFLOPS (Kernel): Our / Torch
    for i, model in enumerate(model_names):
        val_torch = df['torch'][model]
        val_our = df['our'][model]
        
        if val_torch > 0 and val_our > 0:
            if exp_type == 'e2e':
                speedup = val_torch / val_our
            else:
                speedup = val_our / val_torch
            
            # 标注在 Our 的柱子上
            ax.text(x[i] + width, val_our + (val_our * 0.05), f'{speedup:.2f}x', 
                    ha='center', va='bottom', fontweight='bold', color='red')

    ax.set_ylabel(ylabel)
    ax.set_title(f'{title} on RTX 3090')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([MODEL_DISPLAY_NAME[m] for m in model_names])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'repro_fig12_{exp_type}.pdf')
    print(f"Saved repro_fig12_{exp_type}.pdf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./repro_logs')
    args = parser.parse_args()

    data = extract(args.log_dir)
    
    # Plot E2E (Latency)
    plot_chart(data, 'e2e', 'End-to-End Latency (Lower is Better)', 'Time (ms)')
    
    # Plot Kernel (GFLOPS)
    plot_chart(data, 'kernel', 'Kernel Performance (Higher is Better)', 'GFLOPS')