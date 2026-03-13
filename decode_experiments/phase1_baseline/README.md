# Phase 1 Baseline (Revised)

## 目标

为 `decode_experiments` 提供可复现、统计稳健的 `prefill vs decode` baseline（kernel 级）。

## 核心改进

1. 对齐主项目风格：输入构造改为调用模型 `prepare(...)`，兼容额外输入。
2. 统计更稳健：输出 `median/p95/p99/CV`，并支持 `--repeats`。
3. FLOPs 口径更清晰：同时输出 `dense executed` 与 `causal effective`。
4. 带宽口径分离：
   - `io_lower_bound`（输入读 + 输出写）
   - `eager_estimate`（再加中间张量估算）
5. 显存指标更可解释：输出 `peak_delta_mb`，避免把常驻输入误当作阶段开销。
6. 执行路径对齐官方脚本：通过 `compile.py::compile(system=...)` 调用后端（默认 `torch`）。

## 运行

```bash
cd /data2/ldz/FlashTensor-AE/decode_experiments/phase1_baseline
bash run_tests.sh
```

脚本会自动选择空闲显存最多的 GPU。  
如需手动指定，可用：

```bash
DEVICE_INDEX=3 bash run_tests.sh
```

默认 `OOM_POLICY=skip`：当某些长上下文（如 8192）OOM 时，记录该点并继续跑其它点。  
若你希望 OOM 直接失败退出，可用：

```bash
OOM_POLICY=raise bash run_tests.sh
```

单次运行示例：

```bash
python test_prefill_vs_decode.py \
  --model h2o \
  --system torch \
  --context_lengths 2048 4096 8192 \
  --warmup 10 \
  --runs 50 \
  --repeats 3 \
  --dtype float16
```

绘图：

```bash
python plot_results.py --input results/<your_result>.json --output_dir plots
```

## 输出说明

JSON 主要字段：

1. 兼容旧字段：`gpu_avg_time_ms`、`gpu_tflops_per_sec`、`bandwidth_utilization_percent`、`peak_memory_mb`
2. 新增字段：
   - `gpu_median_time_ms` / `gpu_p95_time_ms` / `gpu_cv`
   - `repeat_gpu_cv`（跨 repeat 的稳定性）
   - `flops_dense_gflops` / `flops_causal_gflops`
   - `memory_traffic_eager_estimate_gb`
   - `peak_memory_delta_mb`
   - `system` / `input_names` / `output_names`（可审计执行配置）

## 注意

1. 本实验是 **kernel 级 baseline**，不是完整 LLM 端到端吞吐。
2. 若要做严格跨机对比，请显式设置 `--theoretical_bandwidth_gbs`。
3. 可通过 `--system` 切换后端，建议先用 `--system torch` 建立基准，再扩展到其他系统。
4. 若出现 OOM，先检查 `nvidia-smi` 的空闲显存；当空闲显存不足（例如 < 20GB）时，请切换 GPU 或减少 `--context_lengths`。
5. 计时主口径使用与官方 `perf` 对齐的 `synchronize + wall-clock`；`GPU Event` 作为辅助字段保留在结果中（`gpu_event_*`）。
