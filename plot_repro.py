import sys
import os
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

MODEL_NAMES = ['attn', 'h2o', 'gemma2']

# 模型显示名称
MODEL_DISPLAY_NAME = {
    'attn': 'Standard Attn',
    'h2o': 'H2O (Sparse)',
    'roco': 'Roco',
    'gemma2': 'Gemma2'
}

# 颜色映射（基于系统名，不同 fullgraph 可使用同色不同填充）
COLOR_MAP = {
    'dynamo': '#FFBB78',  # 橙色
    'our': '#98DF8A',      # 绿色
    # 可扩展其他系统
}
HATCH_MAP = {
    'full': '//',          # 全图优化使用斜线填充
    'nofull': ''           # 无全图优化无填充
}

def extract(log_dir):
    """
    从日志文件中提取性能数据。
    文件名格式：{exp}.rtx3090.{model}.{sys}.{fullgraph}.log
    返回: data[exp][(sys, fullgraph)][model] = value
    """
    fn_pattern = r'(?P<exp>[^.]+)\.rtx3090\.(?P<model>[^.]+)\.(?P<sys>[^.]+)\.(?P<fullgraph>[^.]+)\.log'
    time_pattern = re.compile(r'\[.*?\]\s*avg[:\s]+(?P<val>[\d.]+)')
    workload_pattern = re.compile(r'gflops=(?P<val>[\d.]+)')

    # 数据结构：data[exp][(sys, fullgraph)][model] = value
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    if not os.path.exists(log_dir):
        print(f"Error: Log directory {log_dir} not found.")
        return data

    files_found = 0
    for root, _, files in os.walk(log_dir):
        for file in files:
            match = re.match(fn_pattern, file)
            if not match:
                continue

            exp_type = match.group('exp')
            model = match.group('model')
            sys_key = match.group('sys')
            fullgraph = match.group('fullgraph')

            # 只处理我们关心的模型（可选，也可全部保留）
            if model not in MODEL_NAMES:
                continue

            files_found += 1
            fp = os.path.join(root, file)

            with open(fp, 'r') as f:
                content = f.read()

                if exp_type == 'e2e':
                    m = time_pattern.search(content)
                    if m:
                        data[exp_type][(sys_key, fullgraph)][model] = float(m.group('val'))
                else:  # kernel
                    m_time = time_pattern.search(content)
                    m_work = workload_pattern.search(content)
                    if m_time and m_work:
                        latency_ms = float(m_time.group('val'))
                        workload_gflops = float(m_work.group('val'))
                        if latency_ms > 0:
                            throughput = workload_gflops / (latency_ms / 1000.0)
                            data[exp_type][(sys_key, fullgraph)][model] = throughput

    print(f"Parsed {files_found} log files.")
    return data

def plot_results(data, exp_type, plot_title, ylabel):
    """绘制指定实验类型的结果，保留所有 (sys, fullgraph) 组合"""
    if exp_type not in data:
        print(f"No data for {exp_type}")
        return

    # 获取所有唯一的 (sys, fullgraph) 组合，并按系统分组排序（便于图例）
    combos = sorted(data[exp_type].keys(), key=lambda x: (x[0], x[1]))
    if not combos:
        print(f"No combos for {exp_type}")
        return

    # 准备 DataFrame，行索引为模型，列为每个 combo
    df_dict = {}
    for (sys, fg) in combos:
        col_name = f"{sys}_{fg}"
        df_dict[col_name] = [data[exp_type][(sys, fg)].get(m, 0) for m in MODEL_NAMES]

    df = pd.DataFrame(df_dict, index=MODEL_NAMES)

    if df.values.sum() == 0:
        print(f"Skipping {exp_type} due to empty data")
        return

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODEL_NAMES))
    n_bars = len(combos)
    width = 0.8 / n_bars

    for i, (sys, fg) in enumerate(combos):
        col_name = f"{sys}_{fg}"
        offset = width * i - (width * n_bars / 2) + (width / 2)
        vals = df[col_name].values

        # 根据系统和 fullgraph 决定颜色和填充
        color = COLOR_MAP.get(sys, '#CCCCCC')
        hatch = HATCH_MAP.get(fg, '')

        rects = ax.bar(x + offset, vals, width,
                       label=f"{sys} ({fg})",
                       color=color,
                       edgecolor='black',
                       hatch=hatch)

        ax.bar_label(rects, padding=3, fmt='%.1f', fontsize=8)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(plot_title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY_NAME.get(m, m) for m in MODEL_NAMES], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 自动调整 Y 轴上限留出空间
    y_max = df.values.max()
    ax.set_ylim(0, y_max * 1.2 if y_max > 0 else 1)

    plt.tight_layout()
    out_name = f'repro_fig_{exp_type}_all.pdf'
    plt.savefig(out_name)
    print(f"Saved: {out_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./repro_logs')
    args = parser.parse_args()

    all_data = extract(args.log_dir)

    if not all_data:
        print("No data extracted. Please check log filenames and regex.")
        if os.path.exists(args.log_dir):
            print("First few files in log dir:")
            print(os.listdir(args.log_dir)[:5])
    else:
        # 分别绘制 kernel 和 e2e 结果
        plot_results(all_data, 'kernel', 'Kernel Throughput (Higher is Better) (RTX3090)', 'GFLOPS/s')
        plot_results(all_data, 'e2e', 'End-to-End Latency (Lower is Better) (RTX3090)', 'Latency (ms)')