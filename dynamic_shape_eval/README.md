# 2.9-3.3

编写了实验代码`dynamic_shape_evaluator.py`，用来评测FlashTensor是否支持可变参数。

在本目录下执行：（模型和序列长度参数可以改，不影响结论）

`python dynamic_shape_evaluator.py --model attn --base_seqlen 1024 --target_seqlen 2048`

会在`./eval_reports`目录下得到输出的评测结果（md文件），评测中会出现报错：

`RuntimeError: The size of tensor a (2048) must match the size of tensor b (1024) at non-singleton dimension 1`

说明用旧的kernel不能支持新的参数，即FlashTensor不支持可变的参数，只支持静态参数。