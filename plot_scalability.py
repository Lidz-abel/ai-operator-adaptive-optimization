import os
import glob
import re
import matplotlib.pyplot as plt

# 配置
LOG_DIR = "./repro_logs"
MODEL = "h2o"
DEVICE = "rtx3090"

# 数据容器: {seq: {'our': time, 'torch': time}}
data_points = {}

# 正则匹配日志文件名: goal3.rtx3090.h2o.our.seq1024.log
file_pattern = re.compile(r'goal3\.' + DEVICE + r'\.' + MODEL + r'\.(?P<sys>our|torch)\.seq(?P<seq>\d+)\.log')

# 正则匹配执行时间 (从你之前的 plot_repro.py 借用的正则)
time_pattern = re.compile(r'\[(?P<label>our|torch)\].*?avg[:\s]+(?P<val>[\d.]+)')

print(f"Scanning logs in {LOG_DIR}...")

for log_file in glob.glob(os.path.join(LOG_DIR, "goal3*.log")):
    filename = os.path.basename(log_file)
    match = file_pattern.search(filename)
    if match:
        sys_name = match.group('sys')
        seq = int(match.group('seq'))
        
        with open(log_file, 'r') as f:
            content = f.read()
            # 查找时间
            t_match = time_pattern.search(content)
            if t_match:
                latency = float(t_match.group('val'))
                
                if seq not in data_points:
                    data_points[seq] = {}
                data_points[seq][sys_name] = latency
                print(f"Parsed: Seq={seq}, Sys={sys_name} -> Latency={latency} ms")

# 整理数据用于绘图
sorted_seqs = sorted(data_points.keys())
our_times = []
torch_times = []

for s in sorted_seqs:
    our_times.append(data_points[s].get('our', None))
    torch_times.append(data_points[s].get('torch', None))

# === 开始绘图 ===
plt.figure(figsize=(10, 6))

# 画 PyTorch 线
plt.plot(sorted_seqs, torch_times, marker='o', linestyle='--', color='gray', label='PyTorch')

# 画 FlashTensor 线
plt.plot(sorted_seqs, our_times, marker='*', linestyle='-', color='#FF9896', linewidth=2, markersize=10, label='FlashTensor')

plt.title(f'Scalability Analysis: {MODEL} on {DEVICE}', fontsize=14)
plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel('Latency (ms) [Lower is Better]', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(sorted_seqs) # 强制显示 X 轴刻度

output_file = 'repro_fig13_scalability.pdf'
plt.savefig(output_file)
print(f"\nPlot saved to: {output_file}")
