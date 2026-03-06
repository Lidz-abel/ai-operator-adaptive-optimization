#!/usr/bin/env python3

import torch
import sys
import time
import argparse
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Callable

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from compile import compile
    from asuka_exp.cases.kernels import KERNEL_ZOO
except ImportError as e:
    print(f"Warning: Could not import FlashTensor modules: {e}")

class DynamicShapeEvaluator:
    def __init__(self, model_name: str, out_dir: str = "./eval_reports"):
        self.model_name = model_name
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        assert self.model_name in KERNEL_ZOO, f"Model {self.model_name} not found in KERNEL_ZOO."
        
        self.report_data = {
            "metadata": {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": self.model_name},
            "metrics": {},
            "conclusions": {}
        }

    def _prepare_inputs(self, model_inst, seqlen: int, batch: int = 1, dtype: torch.dtype = torch.float16):
        specs = model_inst.prepare(q_len=seqlen, kv_len=seqlen)
        inputs = {}
        for key, tensor in specs['input'].items():
            if isinstance(tensor, torch.Tensor):
                inputs[key] = tensor.clone().to(dtype=dtype).cuda()
        return inputs, specs['output']

    def _measure_sustained_performance(self, kernel: Callable, inputs: Dict, warmup=10, runs=30) -> float:
        # 这里的 warmup 是针对传入的 inputs 做的，因此无论换什么 size 的数据都会正确预热
        input_list = list(inputs.values())
        for _ in range(warmup): 
            kernel(*input_list)
        torch.cuda.synchronize()
        times =[]
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            kernel(*input_list)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return float(np.mean(times))

    def _calc_max_error(self, out1, out2) -> float:
        """递归计算输出结果的最大绝对误差"""
        if isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
            return torch.max(torch.abs(out1.float() - out2.float())).item()
        elif isinstance(out1, (list, tuple)) and isinstance(out2, (list, tuple)):
            err = 0.0
            for t1, t2 in zip(out1, out2):
                err = max(err, self._calc_max_error(t1, t2))
            return err
        elif isinstance(out1, dict) and isinstance(out2, dict):
            err = 0.0
            for k in out1.keys():
                if k in out2:
                    err = max(err, self._calc_max_error(out1[k], out2[k]))
            return err
        return 0.0

    def _export_report(self):
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.out_dir}/eval_{self.model_name}_{timestamp_str}"
        with open(f"{prefix}.json", 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=4, ensure_ascii=False)

        md = f"# FlashTensor 可变参数(Dynamic Shape) 深度探索报告\n\n"
        md += f"**评测时间**: {self.report_data['metadata']['timestamp']} | **评测模型**: `{self.model_name}`\n"
        md += f"**基础参数 (Base)**: seqlen = {self.report_data['metadata'].get('base_seqlen', 'N/A')}\n"
        md += f"**突变参数 (Target)**: seqlen = {self.report_data['metadata'].get('target_seqlen', 'N/A')}\n\n"

        conc = self.report_data['conclusions']
        m = self.report_data['metrics']

        md += "## 1. 核心结论摘要\n"
        if conc.get('hard_crash'):
            md += "- **是否支持参数形变**: ❌ 不支持 (参数突变导致底层硬崩溃，如显存越界)\n"
        elif not conc.get('is_correct'):
            md += "- **是否支持参数形变**: ❌ 不支持 (发生极其危险的**静默计算错误 Silent Failure**)\n"
            md += "- **原由**: 强行复用旧Kernel跑新参数时，未报错但算出了与 PyTorch 真值完全不符的垃圾数据 (数据截断/溢出)！\n"
        else:
            md += "- **是否支持参数形变**: ✔️ 表面支持 (数值正确且未崩溃)\n"
            md += f"- **性能退化**: {'⚠️ 发生明显退化' if conc.get('performance_degraded') else '✅ 性能正常'}\n"

        md += "\n## 2. 数值正确性校验 (Correctness Check vs Eager Mode)\n"
        if conc.get('hard_crash'):
            md += f"- **异常信息**: 运行过程中发生硬崩溃。`{m.get('crash_error', 'N/A')}`\n"
        else:
            md += f"- **旧策略与 PyTorch Eager 真值的最大绝对误差**: {m.get('max_error_vs_eager', 0):.4f}\n"
            md += f"- **新原生策略与 PyTorch Eager 真值的最大绝对误差**: {m.get('native_error_vs_eager', 0):.4f}\n"
            if not conc.get('is_correct'):
                md += "> 🚨 **重大发现**：专门为大尺寸编译的 Native Kernel 是正确的，但强行复用的旧 Kernel 算出了彻底错误的结果！这说明底层 Grid Size、步长或循环边界被静态锁死，遇到了无法兼容的数据跨度。性能对比已无意义。\n"
            else:
                md += "> ✅ 数值校验通过，允许进行性能对比。\n"

        md += "\n## 3. 运行性能对齐测试\n"
        if not conc.get('hard_crash') and conc.get('is_correct'):
            md += f"- **A. 强行复用旧策略的运行耗时**: {m.get('reused_strategy_ms', 0):.3f} ms\n"
            md += f"- **B. 专门为新尺寸重新编译的专属耗时**: {m.get('native_strategy_ms', 0):.3f} ms\n"
            ratio = m.get('reused_strategy_ms', 1) / m.get('native_strategy_ms', 1)
            md += f"- **性能对比**: 强行复用旧策略比原生优化慢了 **{((ratio - 1) * 100):.2f}%**\n"
        else:
            md += "> ⚠️ **跳过对比**：由于系统崩溃或计算结果错误，性能对比已被阻断。\n"

        with open(f"{prefix}.md", 'w', encoding='utf-8') as f: 
            f.write(md)
        print(f"\n[✔️] 报告已导出: {prefix}.md")

    def run_evaluation(self, base_seqlen: int = 1024, target_seqlen: int = 4096):
        self.report_data['metadata'].update({'base_seqlen': base_seqlen, 'target_seqlen': target_seqlen})
        
        try:
            # 0. 实例化并严格同步权重 
            cls = KERNEL_ZOO[self.model_name]
            model_base = cls().eval().cuda()
            model_native = cls().eval().cuda()
            
            model_native.load_state_dict(model_base.state_dict())
            print(f"[*] 成功初始化两份独立的模型对象，并完成权重绝对同步。")

            # 1. 准备 Base 数据并编译
            print(f"▶️ [1/4] 编译 Base Kernel ({base_seqlen}) ...")
            inputs_base, out_names_base = self._prepare_inputs(model_base, base_seqlen)
            kernel_base = compile(
                model=model_base, input_names=list(inputs_base.keys()),
                inputs=list(inputs_base.values()), output_names=out_names_base, system='our'
            )
            del inputs_base
            torch.cuda.empty_cache()

            # 2. 准备 Target 数据并编译原生 Kernel
            print(f"▶️ [2/4] 编译 Target Native Kernel ({target_seqlen}) ...")
            inputs_tgt_native, out_names_target = self._prepare_inputs(model_native, target_seqlen)
            inputs_tgt_reused = {k: v.clone() for k, v in inputs_tgt_native.items()}

            kernel_native = compile(
                model=model_native, input_names=list(inputs_tgt_native.keys()),
                inputs=list(inputs_tgt_native.values()), output_names=out_names_target, system='our'
            )

            
            print("▶️ [3/4] 校验正确性 (对比 PyTorch 原生真值) ...")
            with torch.no_grad():
                # 使用原生 PyTorch forward 获取基准答案
                res_eager = model_native(**inputs_tgt_native)
            
            # 让两个编译出来的 Kernel 也跑一跑
            try:
                res_native = kernel_native(*list(inputs_tgt_native.values()))
                torch.cuda.synchronize()

                res_reused = kernel_base(*list(inputs_tgt_reused.values()))
                torch.cuda.synchronize()
            except Exception as e:
                self.report_data['conclusions']['hard_crash'] = True
                self.report_data['metrics']['crash_error'] = str(e)
                print(f"🚨 警告：执行中发生崩溃！报错: {e}")
                return

            # 对比真值 
            if res_eager is None:
                # Fallback: 如果没有返回值，则比对被修改的特定字典键
                max_err_native = 0.0
                max_err_reused = 0.0
                for out_k in out_names_target:
                    # 假定输出存在于修改后的输入列表中
                    if out_k in inputs_tgt_native:
                        max_err_native = max(max_err_native, self._calc_max_error(inputs_tgt_native[out_k], inputs_tgt_native[out_k])) 
                        max_err_reused = max(max_err_reused, self._calc_max_error(inputs_tgt_native[out_k], inputs_tgt_reused[out_k]))
            else:
                max_err_native = self._calc_max_error(res_eager, res_native)
                max_err_reused = self._calc_max_error(res_eager, res_reused)

            self.report_data['metrics']['native_error_vs_eager'] = max_err_native
            self.report_data['metrics']['max_error_vs_eager'] = max_err_reused

            
            is_correct = max_err_reused < max(1e-3, max_err_native * 10)
            self.report_data['conclusions']['is_correct'] = is_correct

            if not is_correct:
                self.report_data['conclusions']['hard_crash'] = False
                print(f"🚨 警告：旧Kernel算出了完全错误的结果！最大误差: {max_err_reused:.4f} (参考原生误差: {max_err_native:.4f})")
                return 

            # 4. 性能测试
            print("▶️ [4/4] 测算运行性能差距...")
            reused_sus = self._measure_sustained_performance(kernel_base, inputs_tgt_reused)
            native_sus = self._measure_sustained_performance(kernel_native, inputs_tgt_native)
            self.report_data['metrics']['reused_strategy_ms'] = reused_sus
            self.report_data['metrics']['native_strategy_ms'] = native_sus
            self.report_data['conclusions']['performance_degraded'] = (reused_sus / native_sus) > 1.05

        except Exception as e:
            self.report_data['conclusions']['hard_crash'] = True
            self.report_data['metrics']['crash_error'] = f"{type(e).__name__}: {str(e)}"
            print(f"🚨 意外崩溃: {type(e).__name__}: {str(e)}")
            
        finally:
            self._export_report()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FlashTensor 性能与正确性综合评测")
    parser.add_argument('--model', type=str, default='attn')
    parser.add_argument('--base_seqlen', type=int, default=1024)
    parser.add_argument('--target_seqlen', type=int, default=4096)
    args = parser.parse_args()
    
    evaluator = DynamicShapeEvaluator(model_name=args.model)
    evaluator.run_evaluation(base_seqlen=args.base_seqlen, target_seqlen=args.target_seqlen)