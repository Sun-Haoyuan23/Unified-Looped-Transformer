# Show-o2 第一阶段：官方基线与计算剖析

这是研究方案的第一阶段实现。它调用 Show-o2 官方的模型、输入构造、注意力掩码、Euler flow 采样器和 Wan2.1 VAE。脚本保存模型直接生成的 PNG 与逐图日志。当前 14 个开发提示覆盖组合约束、人物、自然、插画和产品，每个提示固定 4 个随机种子，共 56 张图。它们用于工程基线与后续观察实验，不当作论文测试集。

服务器工作目录：`/home/shy_23/iterative_visual_stage1_20261003`。环境：`/home/shy_23/venvs/show_o2/bin/python`。官方仓库固定到运行时记录的 commit。当前分发方式为每张健康 GPU 启动一个独立进程，按提示词分片；同一提示词的四个种子保留在同一进程中。

模型下载：`showlab/show-o2-1.5B` 的 `pytorch_model.bin`，Qwen2.5-1.5B tokenizer/config，SigLIP SO400M，及现有 Wan2.1 VAE。`prepare_checkpoint.py` 通过硬链接复用官方完整权重，并把模型初始化所需的 Qwen/SigLIP 路径指向本地；更改 `load_from_showo=True` 可避免重复下载/载入 Qwen 的 3GB 初始化权重。正式输出前检查 Diffusers 的缺失键日志，确认完整 checkpoint 覆盖模型参数。硬链接没有修改官方权重内容，SHA256 写入清单。

典型操作：

```bash
WORK=/home/shy_23/iterative_visual_stage1_20261003
PY=/home/shy_23/venvs/show_o2/bin/python
$PY "$WORK/prepare_checkpoint.py" --work "$WORK" --siglip "$WORK/models/siglip-so400m"
CUDA_VISIBLE_DEVICES=0 TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$WORK/deps" $PY "$WORK/run_baseline.py" --work "$WORK" --gpu-physical 0 --steps 2 --max-images 1 --smoke
$PY "$WORK/launch_all_gpus.py" --work "$WORK" --steps 50 --guidance 5.0
CUDA_VISIBLE_DEVICES=GPU-UUID TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$WORK/deps" $PY "$WORK/train_probe.py" --work "$WORK" --gpu-physical 0 --steps 200
```

报告字段包括：提示词、种子、采样步数、CFG、图像路径、准备/去噪/解码耗时、峰值显存，以及每张卡首图的语言骨干、语义模块和 flow 头 CUDA event 耗时。带组件探针的首图延迟单列；正式延迟统计其余未插桩图片。GPU 总耗时按所有子模块求和，和墙钟时间含义不同。

服务器已有环境的 `torch.compile` 在导入官方模型时因 `functorch.compile` 缺失而报错。这里设置 `TORCHDYNAMO_DISABLE=1`，使官方代码中被导入但本次 naive attention mask 不使用的 flex-attention 编译入口返回原函数；采样使用的数值路径仍与官方脚本一致。该环境修复计入复现记录。

官方 `utils.py` 导入 `decord`，现有环境缺少该包。本项目将 `decord==0.6.0` 安装在 `$WORK/deps`，通过 `PYTHONPATH` 使用，不修改共享虚拟环境。

服务器有一张 3090 的设备句柄异常，使 CUDA 数字序号与 `nvidia-smi` 的物理编号错位。正式启动器按 `nvidia-smi` 查询每张健康卡的 GPU UUID，并用 UUID 设置 `CUDA_VISIBLE_DEVICES`，避免把物理卡 6 错认成卡 7。第一次正式运行发现并修正了该问题；卡 6、7 的分片随后按 UUID 补跑。

这个阶段验证可复现推理与实际成本，不声称完成质量、偏好或多样性的人类评价。第二阶段在这里保存的中间状态和开发提示上加入观察与同状态干预实验。

`train_probe.py` 用 14 张基线图经官方 VAE 重新编码，按官方流路径构造训练对，在生成头的 q/v 投影上运行 rank-16 LoRA 的 200 步微批量试测。它只测真实前向/反向的吞吐与显存；这批由基础模型自己生成的图像不构成质量或偏好训练集，也不能据此报告方法收益。把 `GPU-UUID` 换成健康卡的实际 UUID，或用 `nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader` 查询。
