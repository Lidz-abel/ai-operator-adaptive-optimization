#!/usr/bin/env python3
"""
Phase 1 baseline: Prefill vs Decode kernel-level profiling.

目标：
1. 作为 decode_experiments 的稳定 baseline，输出可复现实验结果。
2. 统计口径更稳健：增加 median / p95 / repeat 间稳定性指标。
3. 兼容多种 kernel（含额外输入如 corm_mask/exp_rand）。

说明：
- 本脚本是 kernel 级测试，不是完整 LLM 端到端。
- 带宽与 FLOPs 指标包含“定义说明”与“估算口径”，避免误读。
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
# 允许直接从仓库根目录导入 compile.py 与 asuka_exp
sys.path.insert(0, str(PROJECT_ROOT))

from asuka_exp.cases.kernels import KERNEL_ZOO
from asuka_exp.utils import compare
from compile import compile as compile_backend

MODEL_ALIASES = {
    # 兼容历史命名
    "keyformer": "kf",
}

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

SYSTEM_CHOICES = [
    "torch",
    "dynamo",
    "tensorrt",
    "tvm",
    "xla",
    "our",
    "flashinfer",
    "flashattn",
]

OOM_POLICY_CHOICES = ["raise", "skip"]

# 常见显卡理论带宽（GB/s）
KNOWN_THEORETICAL_BW_GBS = {
    "NVIDIA GeForce RTX 3090": 936.0,
    "NVIDIA A100-SXM4-80GB": 2039.0,
    "NVIDIA A100-SXM4-40GB": 1555.0,
    "NVIDIA H100 80GB HBM3": 3350.0,
    "NVIDIA H100 PCIe": 2000.0,
}


def flatten_tensors(x: Any) -> List[torch.Tensor]:
    """将任意嵌套输出结构拍平成 Tensor 列表，便于统一统计输出写流量。"""
    if torch.is_tensor(x):
        return [x]
    if isinstance(x, (tuple, list)):
        out: List[torch.Tensor] = []
        for item in x:
            out.extend(flatten_tensors(item))
        return out
    if isinstance(x, dict):
        out = []
        for item in x.values():
            out.extend(flatten_tensors(item))
        return out
    return []


def summarize_times_ms(times_ms: Iterable[float]) -> Dict[str, float]:
    """给定时间样本（ms），返回稳健统计（含 percentile 与 CV）。"""
    arr = np.array(list(times_ms), dtype=np.float64)
    if arr.size == 0:
        return {
            "avg": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "cv": float("nan"),
        }
    avg = float(np.mean(arr))
    std = float(np.std(arr))
    cv = float(std / avg) if avg > 0 else float("nan")
    return {
        "avg": avg,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": std,
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "cv": cv,
    }


class MemoryBandwidthProfiler:
    """估算流量并计算利用率。"""

    def __init__(self, theoretical_bandwidth_gbs: float):
        self.theoretical_bandwidth_gbs = theoretical_bandwidth_gbs

    @staticmethod
    def _tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    def estimate_memory_traffic(
        self,
        inputs: Dict[str, torch.Tensor],
        outputs: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        返回两套口径：
        1) io_lower_bound: 必然发生的输入读取 + 输出写出
        2) eager_estimate: 在 eager 下中间张量（scores/probs/mask）保守估算读写
        """
        input_read_bytes = sum(self._tensor_nbytes(t) for t in inputs.values())
        output_write_bytes = sum(self._tensor_nbytes(t) for t in outputs)

        scores_rw_bytes = 0
        probs_rw_bytes = 0
        mask_rw_bytes = 0

        if "q" in inputs and "k" in inputs:
            q = inputs["q"]  # [B, q_len, H, D]
            k = inputs["k"]  # [B, kv_len, H, D]
            batch_size = int(q.shape[0])
            q_len = int(q.shape[1])
            head_num = int(q.shape[2])
            kv_len = int(k.shape[1])
            bytes_per_qk = q.element_size()

            # scores 通常与 q/k dtype 同步
            scores_bytes = batch_size * head_num * q_len * kv_len * bytes_per_qk
            # probs 在当前实现里 softmax(scores.float())，故按 fp32 估算
            probs_bytes = batch_size * head_num * q_len * kv_len * 4
            # mask: [1,1,q_len,kv_len]
            mask_bytes = q_len * kv_len * bytes_per_qk

            # 读+写估算
            scores_rw_bytes = 2 * scores_bytes
            probs_rw_bytes = 2 * probs_bytes
            mask_rw_bytes = 2 * mask_bytes

        io_lower_bound_bytes = input_read_bytes + output_write_bytes
        eager_estimate_bytes = (
            io_lower_bound_bytes + scores_rw_bytes + probs_rw_bytes + mask_rw_bytes
        )

        return {
            "input_read_gb": input_read_bytes / (1024 ** 3),
            "output_write_gb": output_write_bytes / (1024 ** 3),
            "scores_rw_gb": scores_rw_bytes / (1024 ** 3),
            "probs_rw_gb": probs_rw_bytes / (1024 ** 3),
            "mask_rw_gb": mask_rw_bytes / (1024 ** 3),
            "io_lower_bound_gb": io_lower_bound_bytes / (1024 ** 3),
            "eager_estimate_gb": eager_estimate_bytes / (1024 ** 3),
        }

    def calc_bandwidth(self, traffic_gb: float, time_ms: float) -> Dict[str, float]:
        time_s = time_ms / 1000.0
        achieved_gbs = traffic_gb / time_s if time_s > 0 else float("inf")
        utilization = (
            achieved_gbs / self.theoretical_bandwidth_gbs * 100
            if self.theoretical_bandwidth_gbs > 0
            else float("nan")
        )
        return {
            "achieved_bandwidth_gbs": float(achieved_gbs),
            "theoretical_bandwidth_gbs": float(self.theoretical_bandwidth_gbs),
            "utilization_percent": float(utilization),
        }


class FLOPsCalculator:
    """提供两种 FLOPs 口径：dense 执行口径 与 causal 有效口径。"""

    @staticmethod
    def _causal_valid_pairs(q_len: int, kv_len: int) -> int:
        # 对应 mask=triu(diagonal=kv_len-q_len+1)
        # 每个 query 有效 key 数: kv_len - (q_len - 1 - i)
        # 求和后: q_len*kv_len - q_len*(q_len-1)/2
        return q_len * kv_len - (q_len * (q_len - 1) // 2)

    @staticmethod
    def calculate_attention_flops(
        batch_size: int,
        q_len: int,
        kv_len: int,
        head_num: int,
        head_dim: int,
    ) -> Dict[str, float]:
        dense_pairs = q_len * kv_len
        valid_pairs = FLOPsCalculator._causal_valid_pairs(q_len, kv_len)

        dense_flops = 4 * batch_size * head_num * dense_pairs * head_dim
        causal_flops = 4 * batch_size * head_num * valid_pairs * head_dim

        return {
            "dense_flops": float(dense_flops),
            "dense_gflops": float(dense_flops / 1e9),
            "causal_flops": float(causal_flops),
            "causal_gflops": float(causal_flops / 1e9),
            "dense_pairs": int(dense_pairs),
            "causal_valid_pairs": int(valid_pairs),
            "definition": (
                "dense_flops=4*B*H*q_len*kv_len*D (当前 eager 实现的 matmul 执行口径); "
                "causal_gflops 为掩码后理论有效口径"
            ),
        }


class PrefillDecodeComparison:
    def __init__(
        self,
        model_name: str = "h2o",
        batch_size: int = 1,
        head_num: int = 32,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda:0",
        system: str = "torch",
        check: bool = False,
        oom_policy: str = "skip",
        seed: int = 0,
        theoretical_bandwidth_gbs: float = 936.0,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for phase1 baseline")

        self.model_name = model_name
        self.batch_size = batch_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.system = system
        self.check = check
        self.oom_policy = oom_policy
        torch.cuda.set_device(self.device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        assert model_name in KERNEL_ZOO, f"Model {model_name} not found"
        model_cls = KERNEL_ZOO[model_name]
        self.model = model_cls(
            kv_head_num=head_num,
            head_num=head_num,
            head_dim=head_dim,
        ).eval().to(self.device)

        self.bandwidth_profiler = MemoryBandwidthProfiler(theoretical_bandwidth_gbs)
        self.flops_calculator = FLOPsCalculator()

    def _synchronize(self) -> None:
        """显式同步，保证计时与显存统计落在同一口径。"""
        torch.cuda.synchronize(self.device)

    def _prepare_inputs(
        self, q_len: int, kv_len: int
    ) -> Tuple[Dict[str, torch.Tensor], List[str], List[torch.Tensor], List[str]]:
        # 由 kernel model 负责生成当前 shape 下的输入规格
        specs = self.model.prepare(
            batch_size=self.batch_size,
            q_len=q_len,
            kv_len=kv_len,
            dtype=self.dtype,
            device=self.device,
        )
        inputs = specs["input"]
        input_names = list(inputs.keys())
        output_names = list(specs["output"])

        # 防御式检查：输入必须是 Tensor，且位于目标 device
        for name in input_names:
            tensor = inputs[name]
            if not torch.is_tensor(tensor):
                raise TypeError(f"Input {name} is not a tensor: {type(tensor)}")
            if tensor.device != self.device:
                inputs[name] = tensor.to(self.device)
        ordered_inputs = [inputs[name] for name in input_names]
        return inputs, input_names, ordered_inputs, output_names

    def _build_runner(
        self,
        input_names: List[str],
        inputs: List[torch.Tensor],
        output_names: List[str],
    ):
        # 后端能力限制：flashinfer/flashattn 仅支持部分模型
        if self.system in {"flashinfer", "flashattn"} and self.model_name not in {"attn", "gemma2"}:
            raise ValueError(
                f"system={self.system} only supports attn/gemma2, but got {self.model_name}"
            )
        return compile_backend(
            model=self.model,
            input_names=input_names,
            inputs=inputs,
            output_names=output_names,
            system=self.system,
        )

    @staticmethod
    def _forward(f, inputs: List[torch.Tensor]):
        # 纯推理场景，关闭 autograd 减少额外开销
        with torch.no_grad():
            return f(*inputs)

    def _maybe_check_outputs(
        self,
        f,
        inputs: List[torch.Tensor],
        output_names: List[str],
        stage: str,
    ) -> None:
        if not self.check:
            return
        print(f"  correctness check ({stage}, system={self.system})...", flush=True)
        with torch.no_grad():
            ref_out = self.model(*inputs)
            out = f(*inputs)
            self._synchronize()
        compare(out, ref_out, output_names)

    def _measure_timing(
        self,
        f,
        inputs: List[torch.Tensor],
        warmup: int,
        runs: int,
        repeats: int,
    ) -> Dict[str, Any]:
        # gpu_times: CUDA Event 计时（设备视角）
        # cpu_times: sync + wall-clock（端到端视角，作为主口径）
        gpu_times: List[float] = []
        cpu_times: List[float] = []
        gpu_repeat_means: List[float] = []
        cpu_repeat_means: List[float] = []

        for rep in range(repeats):
            # 预热：消除首次调度/缓存未命中带来的冷启动噪声
            print(f"  Repeat {rep+1}/{repeats} warmup ({warmup} runs)...", flush=True)
            for _ in range(warmup):
                _ = self._forward(f, inputs)
            self._synchronize()

            # 次口径：GPU Event，不含 host 侧调度时间
            print(f"  Repeat {rep+1}/{repeats} GPU Event timing ({runs} runs)...", flush=True)
            cur_gpu = []
            for _ in range(runs):
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)

                self._synchronize()
                start_evt.record()
                _ = self._forward(f, inputs)
                end_evt.record()
                self._synchronize()

                cur_gpu.append(float(start_evt.elapsed_time(end_evt)))
            gpu_times.extend(cur_gpu)
            gpu_repeat_means.append(float(np.mean(cur_gpu)))

            # 主口径：sync + wall-clock，和 asuka_exp.utils.perf 保持一致
            print(f"  Repeat {rep+1}/{repeats} CPU timing ({runs} runs)...", flush=True)
            cur_cpu = []
            for _ in range(runs):
                self._synchronize()
                tik = time.time()
                _ = self._forward(f, inputs)
                self._synchronize()
                tok = time.time()
                cur_cpu.append((tok - tik) * 1000.0)
            cpu_times.extend(cur_cpu)
            cpu_repeat_means.append(float(np.mean(cur_cpu)))

        gpu_stats = summarize_times_ms(gpu_times)
        cpu_stats = summarize_times_ms(cpu_times)

        gpu_repeat_stats = summarize_times_ms(gpu_repeat_means)
        cpu_repeat_stats = summarize_times_ms(cpu_repeat_means)

        return {
            "gpu": gpu_stats,
            "cpu": cpu_stats,
            "gpu_repeat": gpu_repeat_stats,
            "cpu_repeat": cpu_repeat_stats,
            "gpu_samples": len(gpu_times),
            "cpu_samples": len(cpu_times),
            "gpu_repeat_means_ms": gpu_repeat_means,
            "cpu_repeat_means_ms": cpu_repeat_means,
        }

    def _measure_memory(self, f, inputs: List[torch.Tensor]) -> Dict[str, float]:
        # 清理缓存并重置峰值统计，尽量隔离单次 forward 的增量
        torch.cuda.empty_cache()
        self._synchronize()

        base_alloc = torch.cuda.memory_allocated(self.device)
        base_reserved = torch.cuda.memory_reserved(self.device)

        torch.cuda.reset_peak_memory_stats(self.device)
        _ = self._forward(f, inputs)
        self._synchronize()

        peak_alloc = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)

        return {
            "base_allocated_mb": base_alloc / (1024 ** 2),
            "base_reserved_mb": base_reserved / (1024 ** 2),
            "peak_allocated_mb": peak_alloc / (1024 ** 2),
            "peak_reserved_mb": peak_reserved / (1024 ** 2),
            "peak_allocated_delta_mb": max(0.0, (peak_alloc - base_alloc) / (1024 ** 2)),
            "peak_reserved_delta_mb": max(0.0, (peak_reserved - base_reserved) / (1024 ** 2)),
        }

    def run_single_test(
        self,
        q_len: int,
        kv_len: int,
        stage: str,
        warmup: int = 10,
        runs: int = 50,
        repeats: int = 1,
    ) -> Dict[str, Any]:
        # stage 只允许 prefill/decode 两种，保证输出 JSON 结构稳定
        assert stage in {"prefill", "decode"}
        is_prefill = stage == "prefill"

        try:
            input_dict, input_names, inputs, output_names = self._prepare_inputs(
                q_len=q_len,
                kv_len=kv_len,
            )
            runner = self._build_runner(
                input_names=input_names,
                inputs=inputs,
                output_names=output_names,
            )
            self._maybe_check_outputs(runner, inputs, output_names, stage=stage)

            # 输出用于流量估算（包括多输出 kernel）
            outputs = flatten_tensors(self._forward(runner, inputs))

            # 指标链路：时间 -> 显存 -> 流量 -> FLOPs
            timing = self._measure_timing(runner, inputs, warmup=warmup, runs=runs, repeats=repeats)
            memory = self._measure_memory(runner, inputs)
            traffic = self.bandwidth_profiler.estimate_memory_traffic(input_dict, outputs)
            flops = self.flops_calculator.calculate_attention_flops(
                self.batch_size, q_len, kv_len, self.head_num, self.head_dim
            )
        except torch.cuda.OutOfMemoryError as err:
            if self.oom_policy == "raise":
                raise
            # skip 策略：记录 OOM 并继续跑后续 context，便于得到完整趋势图
            torch.cuda.empty_cache()
            msg = str(err).split("\n")[0]
            print(f"  [OOM] stage={stage}, context_len={kv_len}: {msg}", flush=True)
            return {
                "stage": stage,
                "is_prefill": is_prefill,
                "status": "oom",
                "error": msg,
                "q_len": q_len,
                "kv_len": kv_len,
                "context_len": kv_len,
                "system": self.system,
                "input_names": [],
                "output_names": [],
                "num_outputs": 0,
            }

        # 主口径对齐 asuka_exp.utils.perf：synchronize + wall-clock
        primary_stats = timing["cpu"]
        event_stats = timing["gpu"]

        # 历史字段命名沿用 gpu_*，但这里时间数值来自主口径 primary_stats
        gpu_median = primary_stats["median"]
        gpu_avg = primary_stats["avg"]
        cpu_median = primary_stats["median"]
        cpu_avg = primary_stats["avg"]

        def to_tflops(gflops: float, ms: float) -> float:
            # TFLOPS = GFLOPS / 秒 / 1000
            return gflops / (ms / 1000.0) / 1000.0

        gpu_tflops_dense_median = to_tflops(flops["dense_gflops"], gpu_median)
        gpu_tflops_dense_avg = to_tflops(flops["dense_gflops"], gpu_avg)
        gpu_tflops_causal_median = to_tflops(flops["causal_gflops"], gpu_median)
        cpu_tflops_dense_median = to_tflops(flops["dense_gflops"], cpu_median)

        bw_lower_median = self.bandwidth_profiler.calc_bandwidth(
            traffic["io_lower_bound_gb"], gpu_median
        )
        bw_lower_avg = self.bandwidth_profiler.calc_bandwidth(
            traffic["io_lower_bound_gb"], gpu_avg
        )
        bw_eager_median = self.bandwidth_profiler.calc_bandwidth(
            traffic["eager_estimate_gb"], gpu_median
        )

        result = {
            "stage": stage,
            "is_prefill": is_prefill,
            "status": "ok",
            "q_len": q_len,
            "kv_len": kv_len,
            "context_len": kv_len,
            "system": self.system,
            "input_names": input_names,
            "output_names": output_names,
            "num_outputs": len(outputs),
            # 兼容旧字段（仍保留 avg/min/max/std）
            "gpu_avg_time_ms": float(primary_stats["avg"]),
            "gpu_min_time_ms": float(primary_stats["min"]),
            "gpu_max_time_ms": float(primary_stats["max"]),
            "gpu_std_time_ms": float(primary_stats["std"]),
            "cpu_avg_time_ms": float(primary_stats["avg"]),
            "cpu_min_time_ms": float(primary_stats["min"]),
            "cpu_max_time_ms": float(primary_stats["max"]),
            "cpu_std_time_ms": float(primary_stats["std"]),
            # 新增稳健统计
            "gpu_median_time_ms": float(primary_stats["median"]),
            "gpu_p95_time_ms": float(primary_stats["p95"]),
            "gpu_p99_time_ms": float(primary_stats["p99"]),
            "gpu_cv": float(primary_stats["cv"]),
            "cpu_median_time_ms": float(primary_stats["median"]),
            "cpu_p95_time_ms": float(primary_stats["p95"]),
            "cpu_cv": float(primary_stats["cv"]),
            # 设备事件计时（次要口径，仅供参考）
            "gpu_event_avg_time_ms": float(event_stats["avg"]),
            "gpu_event_median_time_ms": float(event_stats["median"]),
            "gpu_event_p95_time_ms": float(event_stats["p95"]),
            "gpu_event_cv": float(event_stats["cv"]),
            "repeat_gpu_cv": float(timing["gpu_repeat"]["cv"]),
            "repeat_cpu_cv": float(timing["cpu_repeat"]["cv"]),
            "gpu_samples": int(timing["gpu_samples"]),
            "cpu_samples": int(timing["cpu_samples"]),
            "gpu_repeat_means_ms": timing["gpu_repeat_means_ms"],
            "cpu_repeat_means_ms": timing["cpu_repeat_means_ms"],
            # FLOPs 与 TFLOPS
            "base_gflops": float(flops["dense_gflops"]),
            "effective_gflops": float(flops["dense_gflops"]),
            "flops_dense_gflops": float(flops["dense_gflops"]),
            "flops_causal_gflops": float(flops["causal_gflops"]),
            "flops_dense_pairs": int(flops["dense_pairs"]),
            "flops_causal_valid_pairs": int(flops["causal_valid_pairs"]),
            "flops_reason": flops["definition"],
            "gpu_tflops_per_sec": float(gpu_tflops_dense_median),
            "gpu_tflops_per_sec_avg": float(gpu_tflops_dense_avg),
            "gpu_tflops_per_sec_causal": float(gpu_tflops_causal_median),
            "cpu_tflops_per_sec": float(cpu_tflops_dense_median),
            # 带宽（默认展示 lower-bound, median）
            "memory_traffic_gb": float(traffic["io_lower_bound_gb"]),
            "memory_traffic_eager_estimate_gb": float(traffic["eager_estimate_gb"]),
            "memory_breakdown": {
                # 提供拆解字段，方便后续画图与定位瓶颈
                "input_read_gb": float(traffic["input_read_gb"]),
                "output_write_gb": float(traffic["output_write_gb"]),
                "scores_rw_gb": float(traffic["scores_rw_gb"]),
                "probs_rw_gb": float(traffic["probs_rw_gb"]),
                "mask_rw_gb": float(traffic["mask_rw_gb"]),
                "io_lower_bound_gb": float(traffic["io_lower_bound_gb"]),
                "eager_estimate_gb": float(traffic["eager_estimate_gb"]),
            },
            "achieved_bandwidth_gbs": float(bw_lower_median["achieved_bandwidth_gbs"]),
            "theoretical_bandwidth_gbs": float(bw_lower_median["theoretical_bandwidth_gbs"]),
            "bandwidth_utilization_percent": float(bw_lower_median["utilization_percent"]),
            "bandwidth_lower_bound_avg_util_percent": float(bw_lower_avg["utilization_percent"]),
            "bandwidth_eager_estimate_util_percent": float(bw_eager_median["utilization_percent"]),
            # 显存
            "peak_memory_mb": float(memory["peak_allocated_mb"]),
            "peak_memory_reserved_mb": float(memory["peak_reserved_mb"]),
            "peak_memory_delta_mb": float(memory["peak_allocated_delta_mb"]),
            "peak_memory_reserved_delta_mb": float(memory["peak_reserved_delta_mb"]),
            "base_memory_allocated_mb": float(memory["base_allocated_mb"]),
            "base_memory_reserved_mb": float(memory["base_reserved_mb"]),
        }
        return result

    def run_prefill_test(self, context_len: int, **kwargs) -> Dict[str, Any]:
        print(f"\n[Prefill] context_len={context_len}")
        # Prefill: q_len == kv_len == context_len
        return self.run_single_test(
            q_len=context_len,
            kv_len=context_len,
            stage="prefill",
            **kwargs,
        )

    def run_decode_test(self, context_len: int, **kwargs) -> Dict[str, Any]:
        print(f"\n[Decode] context_len={context_len}")
        # Decode: 单步 token，q_len 固定为 1
        return self.run_single_test(
            q_len=1,
            kv_len=context_len,
            stage="decode",
            **kwargs,
        )

    def compare(self, context_lengths: List[int], **kwargs) -> Dict[str, Any]:
        # 整体结果结构：实验配置 + 每个 context 的 prefill/decode 记录
        results: Dict[str, Any] = {
            "model": self.model_name,
            "system": self.system,
            "check": self.check,
            "batch_size": self.batch_size,
            "head_num": self.head_num,
            "head_dim": self.head_dim,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(self.device),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "methodology": {
                "backend_path": "compile.py::compile(system=...)",
                "timing": (
                    "Primary=sync wall-clock (aligned with asuka_exp.utils.perf); "
                    "Secondary=GPU Event; report median/p95/avg with repeats"
                ),
                "flops": "dense executed FLOPs + causal effective FLOPs",
                "bandwidth": "io_lower_bound and eager_intermediate_estimate",
                "memory": "base/peak allocated+reserved with peak delta",
            },
            "prefill_results": [],
            "decode_results": [],
        }

        for context_len in context_lengths:
            print(f"\n{'=' * 80}")
            print(f"Testing context_len = {context_len}")
            print(f"{'=' * 80}")

            prefill_result = self.run_prefill_test(context_len, **kwargs)
            decode_result = self.run_decode_test(context_len, **kwargs)

            results["prefill_results"].append(prefill_result)
            results["decode_results"].append(decode_result)

            self._print_comparison(prefill_result, decode_result)

        return results

    @staticmethod
    def _print_stage_line(name: str, rec: Dict[str, Any]) -> None:
        if rec.get("status", "ok") != "ok":
            print(
                f"{name:<10} status={rec.get('status', 'unknown')} "
                f"context={rec.get('context_len')} "
                f"error={rec.get('error', '')[:120]}"
            )
            return
        print(
            f"{name:<10} "
            f"median={rec['gpu_median_time_ms']:<10.4f}ms "
            f"p95={rec['gpu_p95_time_ms']:<10.4f}ms "
            f"TFLOPS={rec['gpu_tflops_per_sec']:<9.4f} "
            f"BW(lower)={rec['bandwidth_utilization_percent']:<8.2f}% "
            f"BW(eager)={rec['bandwidth_eager_estimate_util_percent']:<8.2f}% "
            f"CV={rec['gpu_cv']:<7.4f}"
        )

    def _print_comparison(self, prefill: Dict[str, Any], decode: Dict[str, Any]) -> None:
        print(f"\n{'-' * 80}")
        print(f"Comparison Summary (context_len={prefill['context_len']})")
        print(f"{'-' * 80}")

        self._print_stage_line("Prefill", prefill)
        self._print_stage_line("Decode", decode)

        if prefill.get("status", "ok") != "ok" or decode.get("status", "ok") != "ok":
            print("Time ratio: N/A due to non-ok stage status")
            return

        ratio_median = prefill["gpu_median_time_ms"] / decode["gpu_median_time_ms"]
        ratio_p95 = prefill["gpu_p95_time_ms"] / decode["gpu_p95_time_ms"]

        print(
            f"Time ratio: median={ratio_median:.2f}x, p95={ratio_p95:.2f}x | "
            f"Prefill peak_delta={prefill['peak_memory_delta_mb']:.2f} MB, "
            f"Decode peak_delta={decode['peak_memory_delta_mb']:.2f} MB"
        )


def resolve_model_name(name: str) -> str:
    """将用户输入模型名归一化到 KERNEL_ZOO 键空间。"""
    normalized = name.lower().strip()
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]
    return normalized


def resolve_theoretical_bw(device_name: str, override: float) -> float:
    """优先使用 CLI 覆盖值，否则按设备名查表，默认回退到 3090。"""
    if override is not None:
        return float(override)
    return float(KNOWN_THEORETICAL_BW_GBS.get(device_name, 936.0))


def main():
    # choices 同时支持 canonical name 和 alias，改善 CLI 易用性
    model_choices = sorted(set(list(KERNEL_ZOO.keys()) + list(MODEL_ALIASES.keys())))

    parser = argparse.ArgumentParser(
        description="Phase 1 baseline: Prefill vs Decode kernel profiling"
    )
    parser.add_argument("--model", "-m", type=str, default="h2o", choices=model_choices)
    parser.add_argument("--system", "-s", type=str, default="torch", choices=SYSTEM_CHOICES)
    parser.add_argument("--context_lengths", "-c", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--head_num", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, choices=sorted(DTYPE_MAP.keys()), default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--check", action="store_true", help="compare backend output with eager torch")
    parser.add_argument(
        "--oom_policy",
        type=str,
        default="skip",
        choices=OOM_POLICY_CHOICES,
        help="raise: exit on OOM; skip: record OOM and continue",
    )
    parser.add_argument("--theoretical_bandwidth_gbs", type=float, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)

    args = parser.parse_args()

    resolved_model = resolve_model_name(args.model)
    if resolved_model not in KERNEL_ZOO:
        raise ValueError(f"Resolved model {resolved_model} not in KERNEL_ZOO")

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for phase1 baseline")

    gpu_name = torch.cuda.get_device_name(torch.device(args.device))
    theoretical_bw = resolve_theoretical_bw(gpu_name, args.theoretical_bandwidth_gbs)

    print("=" * 80)
    print("Phase 1 Baseline: Prefill vs Decode")
    print("=" * 80)
    print(f"Model: {args.model} -> {resolved_model}")
    print(f"System: {args.system}")
    print(f"Context Lengths: {args.context_lengths}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Head Num / Head Dim: {args.head_num} / {args.head_dim}")
    print(f"DType: {args.dtype}")
    print(f"Warmup / Runs / Repeats: {args.warmup} / {args.runs} / {args.repeats}")
    print(f"Check Outputs: {args.check}")
    print(f"OOM Policy: {args.oom_policy}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device} ({gpu_name})")
    print(f"Theoretical BW: {theoretical_bw} GB/s")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 80)

    comparison = PrefillDecodeComparison(
        model_name=resolved_model,
        batch_size=args.batch_size,
        head_num=args.head_num,
        head_dim=args.head_dim,
        dtype=DTYPE_MAP[args.dtype],
        device=args.device,
        system=args.system,
        check=args.check,
        oom_policy=args.oom_policy,
        seed=args.seed,
        theoretical_bandwidth_gbs=theoretical_bw,
    )

    results = comparison.compare(
        context_lengths=args.context_lengths,
        warmup=args.warmup,
        runs=args.runs,
        repeats=args.repeats,
    )

    results["args"] = vars(args)
    results["resolved_model"] = resolved_model
    results["theoretical_bandwidth_gbs"] = theoretical_bw

    if args.output is None:
        # 默认落到 phase1 results 目录，文件名附时间戳防覆盖
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "decode_experiments" / "phase1_baseline" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"prefill_vs_decode_{resolved_model}_{timestamp}.json"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
