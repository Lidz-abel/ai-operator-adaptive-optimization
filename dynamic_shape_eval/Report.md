# FlashTensor 对可变参数的支持与局限性报告

## 脚本功能
为了探索 FlashTensor 对可变参数（一次编译后，输入张量维度发生变化）的支持情况，我使用了一个测试脚本。具体执行流程如下（以输入张量长度 1024，测试张量长度 2048 为例）：
1. **Base 编译**：使用 `seqlen=1024` 的输入张量，让 FlashTensor 走完完整的图优化和编译流程，生成目标函数 `kernel_base`。
2. **获取 Ground Truth**：使用 PyTorch Eager 模式运行 `seqlen=2048` 的输入张量，获取绝对正确的真值输出。
3. **跨尺度强制复用（核心测试）**：将 `seqlen=2048` 的张量直接用到之前为 1024 编译出的 `kernel_base` 执行，不触发重新编译。
4. **正确性与性能判定**：计算复用 Kernel 的输出与 PyTorch 真值的最大绝对误差。只有在数值正确的前提下，才进一步评估其性能退化率；若误差巨大，则判定为发生静默错误，终止性能对比。

##  期望结果与实际结果
- **期望结果**：如果编译器原生支持可变参数，系统应当能够顺利执行 2048 尺寸的输入，并输出与 PyTorch Eager 模式一致的正确结果（即使因为未重新调优导致性能次优）。
- **实际结果**：系统直接发生了硬崩溃，未能执行。
  报错信息为：`RuntimeError: The size of tensor a (2048) must match the size of tensor b (1024) at non-singleton dimension 1`。

## 两者不一致的原因
我推测期望与实际不一致的原因是 FlashTensor 编译生成的执行文件在执行期存在严格的静态形状校验。

当 `kernel_base` 被编译时，`seqlen=1024` 这个具体数值已经被硬编码到底层的执行图中（例如分配中间张量的大小、广播操作的维度等）。当 2048 的张量传入时，底层的 PyTorch 运行时在进行张量计算时，发现实际输入尺寸（2048）与编译期固化的预期尺寸（1024）无法匹配，因此直接抛出了尺寸不匹配的 RuntimeError 异常。

## 结论：FlashTensor 在两个阶段都不支持可变参数
**后续执行阶段**：脚本代码的运行已经说明 FlashTensor 不支持可变参数

**调优阶段**：可以查看源代码中关于`compile`函数的定义：

```python 
>>> import sys
>>> sys.path.insert(0, '/data2/ldz/FlashTensor-AE')
>>> from compile import compile as ft_compile
>>> import inspect
>>> print(inspect.getsource(ft_compile))
```



`compile`函数定义如下：

```python
def compile(model, input_names, inputs, output_names, system):
  if system == 'torch':
    f = model
  elif system == 'dynamo':
    torch._dynamo.reset()
    f = torch.compile(model) 
  elif system == 'tensorrt':
    from asuka_exp.trtllm_utils import trt_build_engine_from_onnx, trt_build_independent_runtime
    onnx_model = torch_module_to_onnx(
      module=model,
      input_names=input_names,
      inputs=inputs,
      output_names=output_names,
    )
    engine = trt_build_engine_from_onnx(onnx_model)
    f = trt_build_independent_runtime(engine)
  elif system == 'xla':
    import torch_xla.core.xla_model as xm
    def _f(*args):
      o = model(*args)
      xm.mark_step()
      xm.wait_device_ops()
      return o
    f = _f
  elif system == 'tvm':
    from asuka_exp.tvm_utils import meta_scheduler_tune, tvm_build_independent_runtime
    lib = meta_scheduler_tune(
      module=model,
      input_names=input_names,
      inputs=inputs,
      output_names=output_names,
      # num_trials_per_iter=64,
      # max_trials_per_task=1000,
      num_trials_per_iter=4,
      max_trials_per_task=128,
      exported_lib_path=None,
    )
    f = tvm_build_independent_runtime(lib, input_names, output_names)
  elif system == 'our':
    from asuka.translate import asuka_from_onnx
    from asuka.transform import fission
    from asuka.transform.common import simplify
    from asuka.partition.connected import Connected
    onnx_model = torch_module_to_onnx(
      module=model,
      input_names=input_names,
      inputs=inputs,
      output_names=output_names,
      simplify=False,
    )
    print(onnx.helper.printable_graph(onnx_model.graph), flush=True)
    func_name = model.__class__.__name__

    import time
    tik = time.time()
    module = asuka_from_onnx(onnx_model, func_name)
    module.dump()
    fission(module)
    simplify(module)
    partition = Connected(module, func_name)
    partition.module.dump()
    partition.optimize()
    tok = time.time()
    tuning_s = tok - tik
    print(f"tuning time: {tuning_s} sec", flush=True)
    perf = partition.profile()
    py_str = partition.codegen(perf)

    our = {}
    import tempfile
    import importlib
    import sys
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
      f.write(py_str)
      path = f.name
    print(f"write code to {path}", flush=True)
    spec = importlib.util.spec_from_file_location('our', path)
    pymod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pymod)

    f = getattr(pymod, func_name)
  elif system == 'flashinfer':
    import flashinfer
    model_name = model.__class__.__name__
    if model_name == 'Attn':
      def _f(*args):
        q, k, v = args[0], args[1], args[2]
        q_len = q.shape[1]
        head_num = q.shape[2]
        head_dim = q.shape[3]
        kv_len = k.shape[1]
        kv_head_num = k.shape[2]
        batch_size = q.shape[0]
        assert batch_size == 1
 
        q = q.view(q_len, head_num, head_dim)
        k = k.view(kv_len, kv_head_num, head_dim)
        v = v.view(kv_len, kv_head_num, head_dim)
        out = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
        return out.view(batch_size, q_len, head_num, head_dim)
    else:
      assert model_name == 'Gemma2'
      def _f(*args):
        q, k, v = args[0], args[1], args[2]
        q_len = q.shape[1]
        head_num = q.shape[2]
        head_dim = q.shape[3]
        kv_len = k.shape[1]
        kv_head_num = k.shape[2]
        batch_size = q.shape[0]
        assert batch_size == 1
 
        q = q.view(q_len, head_num, head_dim)
        k = k.view(kv_len, kv_head_num, head_dim)
        v = v.view(kv_len, kv_head_num, head_dim)
        out = flashinfer.single_prefill_with_kv_cache(q, k, v, logits_soft_cap=50.0, causal=True)
        return out.view(batch_size, q_len, head_num, head_dim)
    f = _f
  elif system == 'flashattn':
    from flash_attn.flash_attn_interface import flash_attn_func
    model_name = model.__class__.__name__
    if model_name == 'Attn':
      def _f(*args):
        q, k, v = args[0], args[1], args[2]
        out = flash_attn_func(q, k, v, causal=True)
        return out
    else:
      assert model_name == 'Gemma2'
      def _f(*args):
        q, k, v = args[0], args[1], args[2]
        out = flash_attn_func(q, k, v, softcap=50.0, causal=True)
        return out
    f = _f
  else:
    raise NotImplementedError(f"system {system} not implemented")
  
  return f
```

可以看出 FlashTensor在调优阶段也不支持可变参数，必须依赖静态的参数。

## 方法本身的局限
我认为 FlashTensor 不支持可变参数是方法本身的局限导致的。

因为根据 FlashTensor 论文：

系统使用模拟退火搜索代数等价图，目标是最小化中间张量的大小。如果 $seqlen$ 是一个可变的符号 $N$，系统就无法计算出具体的内存开销，模拟退火算法直接失去了优化目标。而且该模型需要精确计算“算术强度（Arithmetic Intensity = 计算量 / 访存量）”。如果输入尺寸可变，访存量 `Mem(t)` 就是未知的，系统无法判断哪些算子融合在一起能达到最佳并行度和显存利用率。

