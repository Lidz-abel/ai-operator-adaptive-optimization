# FlashTensor 项目代码讲解（面向 decode_experiments 阶段）

## 1. 文档目标

这份文档用于在 `decode_experiments` 阶段快速建立对 **整个 FlashTensor 项目代码** 的全局认知，重点回答四个问题：

1. 项目有哪些模块。
2. 每个模块怎么实现（核心执行链路）。
3. 各模块有哪些可动参数。
4. 项目里有哪些模型、每个模型做了什么。

---

## 2. 全项目模块地图

从目录功能上，项目可分为 6 层：

1. **编译器核心层（FlashTensor 本体）**：`3rd/asuka`
2. **实验驱动与统一调度层**：`compile.py`、`run_kernel.py`、`run_e2e.py`
3. **算子与模型定义层**：`asuka_exp/cases/kernels`、`asuka_exp/cases/models`
4. **底层 CUDA 扩展层**：`csrc` + `asuka_exp/_csrc`（由 `setup.py` 编译）
5. **论文复现层（fig12/13/14）**：`fig12`、`fig13`、`fig14`、`run_all_repro.sh`
6. **Decode 扩展研究层**：`decode_experiments`（phase1/2/3）

---

## 3. 核心实现链路（从实验到可执行 Kernel）

以 `run_kernel.py` 或 `run_e2e.py` 且 `system=our` 为例，完整链路如下：

1. 构造某个 Kernel 模型（如 `H2O`）并准备输入。
2. 进入 `compile.py::compile(..., system='our')`。
3. `torch.onnx.export` 导出 ONNX。
4. `asuka.translate.asuka_from_onnx`：ONNX -> Asuka IR（MLIR 方言）。
5. `asuka.transform.fission` + `simplify`：先做图变换与规约化。
6. `asuka.partition.connected.Connected`：枚举连通子图并做多阶段剪枝（并行度、AI、合法性等）。
7. `partition.optimize()`：对每个候选 kernel 跑 pass pipeline（并行化、tiling、dynamic for、Asuka->AsukaTriton）。
8. `partition.profile()`：生成临时 Python 并测各 kernel 的时间。
9. `partition.codegen(perf)`：按 profile 结果拼装最终 Python kernel caller。
10. 动态加载临时 `.py`，拿到可调用函数 `f`，用于真实 benchmark / e2e。

这也是 decode_experiments 在分析和改造时要对齐的主线。

---

## 4. 各模块怎么实现

### 4.1 `3rd/asuka`（FlashTensor 编译器核心）

1. **IR 翻译**：`python/asuka/translate.py`
   - 将 ONNX 节点映射到 Asuka IR 操作（MatMul/Softmax/Reduce/Reshape/Permute 等）。
2. **图变换**：`python/asuka/transform`
   - `fission.py`：先拆分复杂 reduce 场景。
   - `common.py`：pass pipeline（simplify、parallelize、tiling、dynamic_for、asuka->asukatriton）。
3. **子图分区与筛选**：`python/asuka/partition`
   - `connected.py`：枚举连通子集，按规则剪枝。
   - `kernel.py`：把 op 子集封装成 kernel。
   - `config.py`：profile + codegen + kernel 调度排序。
4. **代码生成**：`lib/Translate/to_triton.cpp`
   - 把 AsukaTriton IR 生成 Triton Python 代码。

### 4.2 `compile.py`（系统后端统一入口）

支持系统后端：

1. `torch`
2. `dynamo`
3. `tensorrt`
4. `xla`
5. `tvm`
6. `our`（FlashTensor 编译链）
7. `flashinfer`（仅 Attn/Gemma2）
8. `flashattn`（仅 Attn/Gemma2）

`system='our'` 是项目主角；`tvm/tensorrt/dynamo` 用于对比基线。

### 4.3 `asuka_exp/cases/kernels`（Kernel 级算子）

这层是“算子级 forward 定义”，每个类都提供：

1. `forward(...)`：数学实现。
2. `prepare(...)`：构造输入张量规格。
3. `get_model()`：返回类实例。

### 4.4 `asuka_exp/cases/models`（Llama 封装）

这层把 kernel 级 attention 函数 `attn_f` 注入 Llama 结构：

1. 线性层生成 `q/k/v`。
2. 调用 `attn_f`（可来自 torch/tvm/trt/our）。
3. 变体模型按策略生成/筛选 KV cache（TopK、RoCo 统计、SnapKV 平滑等）。
4. 经过 MLP + RMSNorm 输出。

### 4.5 `csrc`（CUDA 扩展）

提供 3 个关键算子给模型层复用：

1. `rms_norm`
2. `silu_and_mul`
3. `rotary_embedding_online`

这些算子在 `llama_base.py` 里被直接调用。

### 4.6 `decode_experiments`（Decode 研究阶段）

#### Phase 1: `phase1_baseline`

1. `test_prefill_vs_decode.py`
   - 以 kernel 级 H2O/Attn/RoCo 做 prefill vs decode 对比。
   - 统计 GPU/CPU 时间、估算 FLOPs、估算带宽与 peak memory。
2. `plot_results.py`
   - 读取 JSON 生成 6 张分析图。

#### Phase 2: `phase2_graph_analysis`

1. `extract_and_analyze_graph.py`
   - 手工定义 H2O 图中 op 序列。
   - 模拟“遇到 reduce 切 kernel”的 Algorithm 1。
   - 输出碎片化得分和融合机会评估。

#### Phase 3: `phase3_fused_kernel`

1. `fused_decode_kernel.py`
   - Triton 手写 fused decode kernel（V2）。
   - 两段式：attention 主计算 + score 归一化回写。
   - 与 baseline 做 correctness + 性能对比。

---

## 5. 各模块可动参数

### 5.1 顶层 `run_kernel.py` 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--model/-m` | `attn` | 选择 kernel 变体 |
| `--system/-s` | `torch` | 选择后端系统 |
| `--seqlen` | `4096` | 序列长度（q_len=kv_len=seqlen） |
| `--show_result` | `False` | 是否打印输出张量 |
| `--check/--no-check` | `True` | 是否与 torch 结果比对 |
| `--fullgraph/--no-fullgraph` | `False` | 预留开关（kernel 实验里基本不使用） |

### 5.2 顶层 `run_e2e.py` 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--model/-m` | `attn` | 选择注意力变体 |
| `--system/-s` | `torch` | 选择后端系统 |
| `--seqlen` | `4096` | 输入序列长度 |
| `--layer_num` | `None` | 覆盖模型层数 |
| `--platform/-p` | `yes` | 权重平台键（读取 `weight_zoo.json`） |
| `--fullgraph/--no-fullgraph` | `False` | 是否对整模型 `torch.compile` |

### 5.3 `compile.py` 中后端相关可调项

| 后端 | 关键可调项 |
|---|---|
| `tvm` | `num_trials_per_iter=4`、`max_trials_per_task=128`（当前写死在代码中） |
| `our` | 分区策略 `Connected` + pass pipeline + profile 驱动 codegen |
| `flashinfer` | 仅支持 `Attn/Gemma2`，且要求 `batch_size==1` |
| `flashattn` | 仅支持 `Attn/Gemma2` |

## 6. 模型清单（Kernel 层 + Llama 层）

项目当前模型键（`KERNEL_ZOO` 与 `MODEL_ZOO` 对齐）：

- `attn`
- `corm`
- `gemma2`
- `h2o`
- `kf`
- `roco`
- `snapkv`

### 6.1 Kernel 层模型含义

| 键名 | 类 | 主要输出 | 特有参数 |
|---|---|---|---|
| `attn` | `Attn` | `out` | `kv_head_num/head_num/head_dim` |
| `corm` | `Corm` | `out, corm_score` | 额外输入 `corm_mask` |
| `gemma2` | `Gemma2` | `out` | `logit_softcapping` |
| `h2o` | `H2O` | `out, h2o_score` | 无额外输入 |
| `kf` | `KeyFormer` | `out, kf_score` | `tau`，额外输入 `exp_rand` |
| `roco` | `RoCo` | `out, roco_score, roco_sq_score` | 无额外输入 |
| `snapkv` | `SnapKV` | `out, snapkv_score` | `kernel_size` |

### 6.2 Llama 层 KV 策略差异（`asuka_exp/cases/models`）

| 键名 | KV 策略 |
|---|---|
| `attn` | 直接拼接 `k/v` 作为 cache |
| `corm` | 使用 `corm_score` 的非零索引记录候选 + 原始 kv |
| `gemma2` | 类似 `attn`，但 attention logits 有 softcapping |
| `h2o` | 对 `h2o_score` 做 TopK，Gather 出压缩 KV |
| `kf` | 基于 `kf_score` TopK，Gather 压缩 KV |
| `roco` | 结合 `roco_score` 和 `roco_sq_score` 做 recent+重要性筛选 |
| `snapkv` | 对 `snapkv_score`（池化后）TopK，Gather 压缩 KV |

---

---

## 7. 备注

1. 本文档基于当前仓库代码状态编写。

