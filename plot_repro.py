import sys
import os
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 定义模型和系统
model_names = ['attn', 'h2o', 'gemma2']
sys_names = ['torch', 'our']

# 配色与阴影
COLOR_DEF = ['#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5', '#C49C94', '#F7B6D2']
HATCH_DEF = [None, '//', '\\\\', 'xx', '..', '++', '**']

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
    # 文件名匹配正则
    fn_pattern = r'(?P<exp>[^.]+)\.rtx3090\.(?P<model>[^.]+)\.(?P<sys>[^.]+)\.log'
    
    # 核心正则 1：匹配执行时间 (支持 [our] avg 0.6 ms 或 [our] avg: 0.6 ms)
    time_pattern = re.compile(r'\[(?P<label>our|torch)\].*?avg[:\s]+(?P<val>[\d.]+)')
    
    # 核心正则 2：匹配理论计算任务量 (gflops=34.36)
    workload_pattern = re.compile(r'gflops=(?P<val>[\d.]+)')

    data = {
        "kernel": {"rtx3090": {sys: {m: 0.0 for m in model_names} for sys in sys_names}},
        "e2e": {"rtx3090": {sys: {m: 0.0 for m in model_names} for sys in sys_names}},
    }

    if not os.path.exists(log_dir):
        print(f"Error: Log directory {log_dir} not found.")
        return data

    for root, _, files in os.walk(log_dir):
        for file in files:
            match = re.match(fn_pattern, file)
            if not match: continue
            
            exp_type = match.group('exp')
            model = match.group('model')
            sys_key = match.group('sys')
            
            if model not in model_names or sys_key not in sys_names: continue
            
            fp = os.path.join(root, file)
            with open(fp, 'r') as f:
                content = f.read()
                
                if exp_type == 'e2e':
                    # E2E 直接提取延迟数值
                    m = time_pattern.search(content)
                    if m:
                        data["e2e"]["rtx3090"][sys_key][model] = float(m.group('val'))
                else:
                    # Kernel 计算吞吐量: GFLOPS/s = (理论计算量) / (耗时s)
                    m_time = time_pattern.search(content)
                    m_work = workload_pattern.search(content)
                    
                    if m_time and m_work:
                        latency_ms = float(m_time.group('val'))
                        workload_gflops = float(m_work.group('val'))
                        # 公式换算
                        throughput = workload_gflops / (latency_ms / 1000.0)
                        data["kernel"]["rtx3090"][sys_key][model] = throughput
                    elif m_time:
                        # 兜底：如果没找到 gflops=，可能是 log 里直接输出了算好的速度
                        # 但在你的 log 里 34.4 是任务量，所以逻辑首选上面的计算
                        data["kernel"]["rtx3090"][sys_key][model] = 0.0 

    return data

def plot_chart(data, exp_type, title, ylabel):
    device = 'rtx3090'
    # 提取数据到 DataFrame
    df = pd.DataFrame(data[exp_type][device])
    
    # 检查是否全为 0
    if df.values.sum() == 0:
        print(f"Warning: No data found for {exp_type}, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_names))
    width = 0.35
    
    # 绘制柱状图
    for i, sys_name in enumerate(sys_names):
        vals = [df[sys_name][m] for m in model_names]
        offset = width * i
        rects = ax.bar(x + offset, vals, width, label=SYS_DISPLAY_NAME[sys_name], 
                       color=COLOR_DEF[i], edgecolor='black', hatch=HATCH_DEF[i])
        
        # 柱子上方的数值标注
        ax.bar_label(rects, padding=3, fmt='%.1f', fontsize=9)

    # 计算并标注加速比 (Speedup)
    for i, model in enumerate(model_names):
        v_torch = df['torch'][model]
        v_our = df['our'][model]
        
        if v_torch > 0 and v_our > 0:
            if exp_type == 'e2e':
                speedup = v_torch / v_our # 延迟越低越好
            else:
                speedup = v_our / v_torch # 吞吐越高越好
            
            # 标注在 FlashTensor (橙色) 柱子上方
            ax.text(x[i] + width, max(v_torch, v_our) * 1.05, f'{speedup:.2f}x', 
                    ha='center', va='bottom', fontweight='bold', color='red', fontsize=12)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{title} (RTX 3090)', fontsize=14)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([MODEL_DISPLAY_NAME[m] for m in model_names], fontsize=11)
    ax.legend(fontsize=10)
    
    # 留出余量给红色的 Speedup 文字
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    out_name = f'repro_fig12_{exp_type}.pdf'
    plt.savefig(out_name)
    print(f"Successfully saved {out_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./repro_logs')
    args = parser.parse_args()

    # 提取
    all_data = extract(args.log_dir)
    
    # 绘图 1: 端到端延迟 (越低越好)
    plot_chart(all_data, 'e2e', 'End-to-End Latency (Lower is Better)', 'Latency (ms)')
    
    # 绘图 2: Kernel 吞吐量 (越高越好)
    plot_chart(all_data, 'kernel', 'Kernel Throughput (Higher is Better)', 'GFLOPS/s')