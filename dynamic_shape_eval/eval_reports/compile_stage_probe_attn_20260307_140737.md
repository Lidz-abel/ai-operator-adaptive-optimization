# FlashTensor 调优阶段可变参数支持探测报告

**时间**: 2026-03-07 14:07:19 | **模型**: `attn`

## 总体结论
> ❌ **调优阶段不支持可变参数**：compile接口仅接受单一固定shape输入，编译产物不含任何动态dispatch机制。

## 探测A：symbolic shape输入
- 结论: `compile_crashed`
- 详情: compile崩溃: RuntimeError: The size of tensor a (512) must match the size of tensor b (1024) at non-singleton dimension 1
- compile接口参数: `['model', 'input_names', 'inputs', 'output_names', 'system']`
- 动态相关参数: `[]`

## 探测B：多shape批量编译
- 结论: `interface_rejected`
- 详情: compile不接受多shape列表: Attn.forward() takes 4 positional arguments but 5 were given

## 探测C：编译产物结构
- 结论: `no_dynamic_mechanism_found`
- 详情: 产物结构简单，未发现dispatch或多kernel机制
- dispatch相关属性: `[]`
- 产物类型: `function` (来自 `builtins`)

