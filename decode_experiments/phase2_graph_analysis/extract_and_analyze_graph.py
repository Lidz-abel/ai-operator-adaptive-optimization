#!/usr/bin/env python3
"""
Phase 2 (revised): strict and fair graph-level analysis for decode vs prefill.

What is fixed compared with the old script:
1) No hardcoded latency/speedup claims.
2) No hardcoded "15us launch overhead" assumption.
3) Explicit analysis scope:
   - attention_only: kernel-level H2O path (asuka_exp/cases/kernels/h2o.py)
   - attention_plus_cache: model-level H2O cache selection extension
4) Same estimation rules and kernel-boundary policy are applied to prefill/decode.
5) Output includes full assumptions for auditability.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Dict, List, Sequence


DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}


@dataclass(frozen=True)
class ShapeConfig:
    batch_size: int
    q_len: int
    kv_len: int
    head_num: int
    kv_head_num: int
    head_dim: int
    cache_budget: int
    dtype: str
    compute_intensity_threshold: float

    @property
    def dtype_bytes(self) -> int:
        return DTYPE_BYTES[self.dtype]

    @property
    def fp32_bytes(self) -> int:
        return 4

    @property
    def selected_len(self) -> int:
        return min(self.cache_budget, self.kv_len)


@dataclass(frozen=True)
class OpSpec:
    name: str
    op_type: str
    has_reduce: bool = False
    has_dynamic_shape: bool = False
    has_irregular_access: bool = False
    scope: str = "attention_only"
    note: str = ""


@dataclass
class OpEstimate:
    name: str
    op_type: str
    has_reduce: bool
    has_dynamic_shape: bool
    has_irregular_access: bool
    estimated_flops: float
    estimated_read_bytes: float
    estimated_write_bytes: float
    estimated_total_bytes: float
    arithmetic_intensity: float
    intensity_label: str
    note: str


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def _common_sizes(cfg: ShapeConfig) -> Dict[str, int]:
    b = cfg.batch_size
    q = cfg.q_len
    k = cfg.kv_len
    h = cfg.head_num
    kvh = cfg.kv_head_num
    d = cfg.head_dim
    sel = cfg.selected_len
    return {
        "mask": _numel((1, 1, q, k)),
        "q": _numel((b, q, h, d)),
        "k": _numel((b, k, kvh, d)),
        "v": _numel((b, k, kvh, d)),
        "q_t": _numel((b, h, q, d)),
        "k_t": _numel((b, kvh, k, d)),
        "v_t": _numel((b, kvh, k, d)),
        "scores": _numel((b, h, q, k)),
        "out_t": _numel((b, h, q, d)),
        "out": _numel((b, q, h, d)),
        "h2o_score": _numel((b, kvh, k)),
        "selected_idx": _numel((b, kvh, sel)),
        "gather_out": _numel((b, sel, kvh, d)),
        "kv_cache": _numel((b, 2, sel, kvh, d)),
    }


def _intensity_label(flops: float, total_bytes: float, threshold: float) -> str:
    if total_bytes <= 0:
        return "meta"
    if flops <= 0:
        return "memory-leaning"
    ai = flops / total_bytes
    return "compute-leaning" if ai >= threshold else "memory-leaning"


def _estimate_op(spec: OpSpec, cfg: ShapeConfig) -> OpEstimate:
    sz = _common_sizes(cfg)
    bpe = cfg.dtype_bytes
    fp32 = cfg.fp32_bytes

    flops = 0.0
    read_bytes = 0.0
    write_bytes = 0.0

    if spec.op_type == "mask_create":
        write_bytes = sz["mask"] * bpe
    elif spec.op_type == "triu":
        flops = float(sz["mask"])
        read_bytes = sz["mask"] * bpe
        write_bytes = sz["mask"] * bpe
    elif spec.op_type == "transpose_q":
        read_bytes = sz["q"] * bpe
        write_bytes = sz["q_t"] * bpe
    elif spec.op_type == "transpose_k":
        read_bytes = sz["k"] * bpe
        write_bytes = sz["k_t"] * bpe
    elif spec.op_type == "transpose_v":
        read_bytes = sz["v"] * bpe
        write_bytes = sz["v_t"] * bpe
    elif spec.op_type == "matmul_qk":
        # [B, H, Q, D] x [B, H, D, K] -> [B, H, Q, K]
        flops = float(2 * cfg.batch_size * cfg.head_num * cfg.q_len * cfg.kv_len * cfg.head_dim)
        read_bytes = (sz["q_t"] + sz["k_t"]) * bpe
        write_bytes = sz["scores"] * bpe
    elif spec.op_type == "div_scale":
        flops = float(sz["scores"])
        read_bytes = sz["scores"] * bpe
        write_bytes = sz["scores"] * bpe
    elif spec.op_type == "add_mask":
        flops = float(sz["scores"])
        # Mask is broadcast over batch/head; keep logical read for fairness.
        read_bytes = (sz["scores"] + sz["mask"]) * bpe
        write_bytes = sz["scores"] * bpe
    elif spec.op_type == "cast_scores_fp32":
        read_bytes = sz["scores"] * bpe
        write_bytes = sz["scores"] * fp32
    elif spec.op_type == "softmax":
        # Approximation: max/sub/exp/sum/div ~= 5 FLOPs per element.
        flops = float(5 * sz["scores"])
        read_bytes = sz["scores"] * fp32
        write_bytes = sz["scores"] * fp32
    elif spec.op_type == "cast_probs":
        read_bytes = sz["scores"] * fp32
        write_bytes = sz["scores"] * bpe
    elif spec.op_type == "matmul_pv":
        # [B, H, Q, K] x [B, H, K, D] -> [B, H, Q, D]
        flops = float(2 * cfg.batch_size * cfg.head_num * cfg.q_len * cfg.kv_len * cfg.head_dim)
        read_bytes = (sz["scores"] + sz["v_t"]) * bpe
        write_bytes = sz["out_t"] * bpe
    elif spec.op_type == "transpose_out":
        read_bytes = sz["out_t"] * bpe
        write_bytes = sz["out"] * bpe
    elif spec.op_type == "contiguous_out":
        read_bytes = sz["out"] * bpe
        write_bytes = sz["out"] * bpe
    elif spec.op_type == "sum_h2o_score":
        # reduce over q_len dimension
        flops = float(cfg.batch_size * cfg.kv_head_num * cfg.kv_len * max(cfg.q_len - 1, 0))
        read_bytes = sz["scores"] * fp32
        write_bytes = sz["h2o_score"] * fp32
    elif spec.op_type == "view_out":
        pass
    elif spec.op_type == "view_h2o":
        pass
    elif spec.op_type == "topk_selection":
        # Heuristic complexity proxy for selection.
        flops = float(
            cfg.batch_size
            * cfg.kv_head_num
            * cfg.kv_len
            * max(math.log2(max(cfg.selected_len, 2)), 1.0)
        )
        read_bytes = sz["h2o_score"] * fp32
        write_bytes = sz["selected_idx"] * 8  # torch.topk index output is int64
    elif spec.op_type == "sort_selected":
        flops = float(
            cfg.batch_size
            * cfg.kv_head_num
            * cfg.selected_len
            * max(math.log2(max(cfg.selected_len, 2)), 1.0)
        )
        read_bytes = sz["selected_idx"] * 8
        write_bytes = sz["selected_idx"] * 8
    elif spec.op_type == "expand_selected":
        pass
    elif spec.op_type == "gather_k":
        read_bytes = sz["gather_out"] * bpe + sz["gather_out"] * 8
        write_bytes = sz["gather_out"] * bpe
    elif spec.op_type == "gather_v":
        read_bytes = sz["gather_out"] * bpe + sz["gather_out"] * 8
        write_bytes = sz["gather_out"] * bpe
    elif spec.op_type == "concat_kv_cache":
        read_bytes = 2 * sz["gather_out"] * bpe
        write_bytes = sz["kv_cache"] * bpe
    elif spec.op_type == "view_kv_cache":
        pass
    else:
        raise ValueError(f"Unsupported op_type: {spec.op_type}")

    total_bytes = read_bytes + write_bytes
    ai = (flops / total_bytes) if total_bytes > 0 else 0.0
    return OpEstimate(
        name=spec.name,
        op_type=spec.op_type,
        has_reduce=spec.has_reduce,
        has_dynamic_shape=spec.has_dynamic_shape,
        has_irregular_access=spec.has_irregular_access,
        estimated_flops=flops,
        estimated_read_bytes=read_bytes,
        estimated_write_bytes=write_bytes,
        estimated_total_bytes=total_bytes,
        arithmetic_intensity=ai,
        intensity_label=_intensity_label(flops, total_bytes, cfg.compute_intensity_threshold),
        note=spec.note,
    )


def _attention_only_ops() -> List[OpSpec]:
    return [
        OpSpec("make_mask", "mask_create"),
        OpSpec("causal_triu", "triu"),
        OpSpec("transpose_q", "transpose_q", has_irregular_access=False),
        OpSpec("transpose_k", "transpose_k", has_irregular_access=False),
        OpSpec("transpose_v", "transpose_v", has_irregular_access=False),
        OpSpec("matmul_qk", "matmul_qk"),
        OpSpec("div_scale", "div_scale"),
        OpSpec("add_mask", "add_mask"),
        OpSpec("cast_scores_fp32", "cast_scores_fp32"),
        OpSpec("softmax", "softmax", has_reduce=True),
        OpSpec("cast_probs", "cast_probs"),
        OpSpec("matmul_pv", "matmul_pv"),
        OpSpec("transpose_out", "transpose_out"),
        OpSpec("contiguous_out", "contiguous_out"),
        OpSpec("sum_h2o_score", "sum_h2o_score", has_reduce=True),
        OpSpec("view_out", "view_out"),
        OpSpec("view_h2o", "view_h2o"),
    ]


def _attention_plus_cache_ops() -> List[OpSpec]:
    ops = list(_attention_only_ops())
    ops.extend(
        [
            OpSpec(
                "topk_selection",
                "topk_selection",
                has_reduce=True,
                has_dynamic_shape=True,
                has_irregular_access=True,
                scope="attention_plus_cache",
                note="k is runtime configurable; selection is data-dependent",
            ),
            OpSpec(
                "sort_selected",
                "sort_selected",
                has_irregular_access=True,
                scope="attention_plus_cache",
            ),
            OpSpec(
                "expand_selected",
                "expand_selected",
                scope="attention_plus_cache",
            ),
            OpSpec(
                "gather_k",
                "gather_k",
                has_irregular_access=True,
                scope="attention_plus_cache",
            ),
            OpSpec(
                "gather_v",
                "gather_v",
                has_irregular_access=True,
                scope="attention_plus_cache",
            ),
            OpSpec(
                "concat_kv_cache",
                "concat_kv_cache",
                scope="attention_plus_cache",
            ),
            OpSpec(
                "view_kv_cache",
                "view_kv_cache",
                scope="attention_plus_cache",
            ),
        ]
    )
    return ops


def _get_ops(scope: str) -> List[OpSpec]:
    if scope == "attention_only":
        return _attention_only_ops()
    if scope == "attention_plus_cache":
        return _attention_plus_cache_ops()
    raise ValueError(f"Unknown scope: {scope}")


def _pack_kernels(
    op_estimates: Sequence[OpEstimate], barrier_fn: Callable[[OpEstimate], bool]
) -> List[List[str]]:
    kernels: List[List[str]] = []
    current: List[str] = []
    for op in op_estimates:
        if barrier_fn(op):
            if current:
                kernels.append(current)
                current = []
            kernels.append([op.name])  # barrier op stands alone
            continue
        current.append(op.name)
    if current:
        kernels.append(current)
    return kernels


def _kernel_policies() -> Dict[str, Callable[[OpEstimate], bool]]:
    return {
        # Similar spirit to current Algorithm-1 style split-by-reduce.
        "reduce_barrier": lambda op: op.has_reduce,
        # More conservative decode policy for irregular accesses.
        "reduce_or_irregular_barrier": lambda op: op.has_reduce or op.has_irregular_access,
        # Strictest policy: reduce + dynamic shape + irregular access.
        "strict_decode_barrier": lambda op: (
            op.has_reduce or op.has_dynamic_shape or op.has_irregular_access
        ),
    }


def _analyze_scope(scope: str, cfg: ShapeConfig) -> Dict:
    specs = _get_ops(scope)
    ops = [_estimate_op(spec, cfg) for spec in specs]

    total_flops = sum(op.estimated_flops for op in ops)
    total_read = sum(op.estimated_read_bytes for op in ops)
    total_write = sum(op.estimated_write_bytes for op in ops)
    total_bytes = total_read + total_write
    global_ai = (total_flops / total_bytes) if total_bytes > 0 else 0.0

    policies = {}
    for policy_name, barrier_fn in _kernel_policies().items():
        kernels = _pack_kernels(ops, barrier_fn)
        covered_ops = sum(len(k) for k in kernels)
        policies[policy_name] = {
            "kernel_count": len(kernels),
            "covered_ops": covered_ops,
            "coverage_ok": covered_ops == len(ops),
            "kernels": kernels,
        }

    return {
        "scope": scope,
        "shape": asdict(cfg),
        "summary": {
            "total_ops": len(ops),
            "reduce_ops": sum(1 for op in ops if op.has_reduce),
            "dynamic_shape_ops": sum(1 for op in ops if op.has_dynamic_shape),
            "irregular_access_ops": sum(1 for op in ops if op.has_irregular_access),
            "estimated_total_flops": total_flops,
            "estimated_total_read_bytes": total_read,
            "estimated_total_write_bytes": total_write,
            "estimated_total_rw_bytes": total_bytes,
            "global_arithmetic_intensity": global_ai,
            "compute_intensity_threshold": cfg.compute_intensity_threshold,
        },
        "kernel_policies": policies,
        "operators": [asdict(op) for op in ops],
    }


def _ratio(a: float, b: float) -> float:
    return a / b if b != 0 else float("inf")


def _compare(prefill: Dict, decode: Dict) -> Dict:
    ps = prefill["summary"]
    ds = decode["summary"]
    return {
        "prefill_vs_decode_flops_ratio": _ratio(
            ps["estimated_total_flops"], ds["estimated_total_flops"]
        ),
        "prefill_vs_decode_rw_bytes_ratio": _ratio(
            ps["estimated_total_rw_bytes"], ds["estimated_total_rw_bytes"]
        ),
        "prefill_vs_decode_global_ai_ratio": _ratio(
            ps["global_arithmetic_intensity"], ds["global_arithmetic_intensity"]
        ),
    }


def _to_gb(v: float) -> float:
    return v / (1024 ** 3)


def _print_scope_report(scope: str, prefill: Dict, decode: Dict) -> None:
    ps = prefill["summary"]
    ds = decode["summary"]
    cmp_metrics = _compare(prefill, decode)

    print("\n" + "=" * 84)
    print(f"Scope: {scope}")
    print("=" * 84)
    print(f"{'Metric':<40s} | {'Prefill':>16s} | {'Decode':>16s} | {'Ratio':>10s}")
    print("-" * 84)
    print(
        f"{'Total ops':<40s} | {ps['total_ops']:>16d} | {ds['total_ops']:>16d} | "
        f"{_ratio(ps['total_ops'], ds['total_ops']):>10.2f}x"
    )
    print(
        f"{'Estimated FLOPs':<40s} | {ps['estimated_total_flops']:>16.3e} | "
        f"{ds['estimated_total_flops']:>16.3e} | {cmp_metrics['prefill_vs_decode_flops_ratio']:>10.2f}x"
    )
    print(
        f"{'Estimated RW bytes (GB)':<40s} | {_to_gb(ps['estimated_total_rw_bytes']):>16.4f} | "
        f"{_to_gb(ds['estimated_total_rw_bytes']):>16.4f} | {cmp_metrics['prefill_vs_decode_rw_bytes_ratio']:>10.2f}x"
    )
    print(
        f"{'Global arithmetic intensity (FLOPs/Byte)':<40s} | {ps['global_arithmetic_intensity']:>16.4f} | "
        f"{ds['global_arithmetic_intensity']:>16.4f} | {cmp_metrics['prefill_vs_decode_global_ai_ratio']:>10.2f}x"
    )

    for policy in ("reduce_barrier", "reduce_or_irregular_barrier", "strict_decode_barrier"):
        pk = prefill["kernel_policies"][policy]["kernel_count"]
        dk = decode["kernel_policies"][policy]["kernel_count"]
        print(f"{f'Kernel count ({policy})':<40s} | {pk:>16d} | {dk:>16d} | {_ratio(pk, dk):>10.2f}x")

    print("-" * 84)
    print(
        "Notes: no hardcoded kernel-launch or end-to-end latency assumptions are used in this phase."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict/fair graph analysis for FlashTensor decode migration."
    )
    parser.add_argument(
        "--scope",
        choices=["attention_only", "attention_plus_cache", "both"],
        default="both",
        help="Analysis scope. 'both' runs both scopes.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--q_len_prefill", type=int, default=4096)
    parser.add_argument("--q_len_decode", type=int, default=1)
    parser.add_argument("--kv_len", type=int, default=4096)
    parser.add_argument("--head_num", type=int, default=32)
    parser.add_argument("--kv_head_num", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--cache_budget", type=int, default=512)
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES.keys()), default="float16")
    parser.add_argument(
        "--compute_intensity_threshold",
        type=float,
        default=8.0,
        help="Threshold used for per-op compute/memory leaning classification.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--print_ops",
        action="store_true",
        help="Print operator-level estimates for audit/debug.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scopes = (
        ["attention_only", "attention_plus_cache"]
        if args.scope == "both"
        else [args.scope]
    )

    config_common = {
        "batch_size": args.batch_size,
        "kv_len": args.kv_len,
        "head_num": args.head_num,
        "kv_head_num": args.kv_head_num,
        "head_dim": args.head_dim,
        "cache_budget": args.cache_budget,
        "dtype": args.dtype,
        "compute_intensity_threshold": args.compute_intensity_threshold,
    }
    prefill_cfg = ShapeConfig(q_len=args.q_len_prefill, **config_common)
    decode_cfg = ShapeConfig(q_len=args.q_len_decode, **config_common)

    output = {
        "analysis_version": "phase2_v2_strict_fair",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "methodology": {
            "fairness_rules": [
                "prefill/decode use identical operator schema within each scope",
                "no hardcoded launch-overhead or latency-based speedup claims",
                "all metrics are produced by transparent formulas in this script",
                "kernel boundaries are computed by explicit policy functions",
            ],
            "notes": [
                "FLOPs/bytes for topk/sort are heuristic proxies, flagged as such in op notes.",
                "This phase is graph-cost analysis, not kernel microbenchmark.",
            ],
        },
        "config": {
            "scope": args.scope,
            "prefill": asdict(prefill_cfg),
            "decode": asdict(decode_cfg),
        },
        "scopes": {},
    }

    print("\n" + "#" * 84)
    print("# Phase 2 Revised: Strict & Fair Graph Analysis")
    print("#" * 84)
    print(f"Scopes: {', '.join(scopes)}")
    print(
        f"Shape config: B={args.batch_size}, Q(prefill/decode)=({args.q_len_prefill}/{args.q_len_decode}), "
        f"K={args.kv_len}, H={args.head_num}, KVH={args.kv_head_num}, D={args.head_dim}, "
        f"cache_budget={args.cache_budget}, dtype={args.dtype}"
    )

    for scope in scopes:
        prefill = _analyze_scope(scope, prefill_cfg)
        decode = _analyze_scope(scope, decode_cfg)
        cmp_metrics = _compare(prefill, decode)
        output["scopes"][scope] = {
            "prefill": prefill,
            "decode": decode,
            "comparison": cmp_metrics,
        }

        _print_scope_report(scope, prefill, decode)

        if args.print_ops:
            print("\nOperator-level estimates:")
            for op in decode["operators"]:
                print(
                    f"  {op['name']:<20s} "
                    f"flops={op['estimated_flops']:.3e} "
                    f"rw_bytes={op['estimated_total_bytes']:.3e} "
                    f"AI={op['arithmetic_intensity']:.3e} "
                    f"{op['intensity_label']}"
                )

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.tag}_" if args.tag else ""
    out_file = os.path.join(args.output_dir, f"{prefix}graph_analysis_{ts}.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nSaved:", out_file)


if __name__ == "__main__":
    main()
