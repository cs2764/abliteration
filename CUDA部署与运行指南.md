# CUDA 版本部署与运行指南

本项目原生支持基于 NVIDIA GPU (CUDA) 的加速计算。本指南详细说明了如何在配备了显卡的 Linux 或 Windows 机器上，独立配置并运行模型的消融（Abliteration）和推理。

## 1. 环境准备与虚拟环境 (重要)

为保证项目环境与其他应用的隔离，并且遵照系统要求：**严禁改动或删除现有的虚拟环境及启动文档！**。

### 1.1 依赖管理器
确保您的系统已安装了最新的 `uv`（极速 Python 包管理及虚拟环境工具）。

### 1.2 创建与激活本地虚拟环境
项目根目录下，使用 `uv` 创建独立运行环境：
```bash
uv venv .venv
```

**激活该环境** (您必须保证每次运行前都在该虚拟环境中)：
- **Linux**:
  ```bash
  source .venv/bin/activate
  ```
- **Windows**:
  ```cmd
  .venv\Scripts\activate
  ```

### 1.3 安装含 CUDA 支持的依赖库
在已经激活 `.venv` 的情况下，运行以下命令（它会自动通过 `requirements.txt` 内的定义拉取 `torch` 等工具对应的 CUDA 兼容版本）：
```bash
uv pip install -r requirements.txt
```

如果您的机器需要额外或指定的 `torch` CUDA 版本（例如 cu121/cu118），请单独指定：
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 2. 下载测试模型

最新版本（v1.5.0及以上）的 `huggingface_hub` 提供了全新的 `hf` 命令行工具。
如果还没安装，请执行：
```bash
uv pip install -U huggingface_hub
```

**下载模型**（以 `Qwen/Qwen3.5-35B-A3B` 为例）：
```bash
hf download Qwen/Qwen3.5-35B-A3B --local-dir ./models/Qwen3.5-35B-A3B
```

## 3. CUDA 专用配置

为了针对显卡运行做出最好优化，项目提供了一份独立的 `config_cuda.yaml` 文件。

你可以直接使用该文件或基于其实例修改：
```yaml
model: "./models/Qwen3.5-35B-A3B"
output_dir: "./models/Qwen3.5-35B-A3B-abliterated-cuda"

inference:
  device: "cuda"  # 将引擎显式指定为 CUDA
  batch_size: 2
  max_lengh: 512
  flash_attn: true # 推荐在 Ampere 架构（例如 RTX3090, A100）或更高版本的机器上开启，以节省显存。
```
**注意：** 如果你的 GPU 不支持 `flash_attention_2`（比如较老的 V100 等显卡），请将 `flash_attn` 设置为 `false`，否则运行会产生报错。

## 4. 开始运行

确认配置文件和底层显卡一切就绪：

### 4.1 运行模型消融
指定专用的 CUDA 配置文件来启动处理程序：
```bash
python abliterate.py config_cuda.yaml
```
程序运行时将自动将所需张量转移到 GPU (CUDA) 并执行直方图路由、矩阵投影等大规模加速运算。

### 4.2 对话体验与对比
模型完成转换后，可以加载新模型进行聊天：
```bash
python chat.py -m ./models/Qwen3.5-35B-A3B-abliterated-cuda
```

如果您想对比前后的表现：
```bash
python compare.py -a ./models/Qwen3.5-35B-A3B -b ./models/Qwen3.5-35B-A3B-abliterated-cuda
```

---
**特别注意事项（请务必遵守）：**
- **严禁**：私自改动本虚拟环境(`.venv`)。
- **严禁**：删除本 CUDA 部署与启动文档、`启动说明.md`。
- 本地生成的所有的敏感配置信息或生成后的模型，不得被上传至 GitHub 仓库。
