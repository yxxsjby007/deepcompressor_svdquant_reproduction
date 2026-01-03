# DeepCompressor 算法复现实验

本项目是对 MIT HAN Lab 开源项目 [DeepCompressor](https://github.com/mit-han-lab/deepcompressor) 的算法复现实验，主要复现了 **SVDQuant** 算法在扩散模型上的4位量化效果。
## 项目目录结构
```
deepcompressor/
├── README.md                   # 本文件
├── outputs_example             # 图片生成实例
├── examples/
│   └── diffusion/              # 扩散模型量化实验results
│       ├── README.md           # 扩散模型实验详细说明
│       ├── configs/            # 配置文件
│       ├── baselines/          # Baseline实验结果
│       └── runs/               # 量化实验结果
├── tools/
│   ├── checkprompt.py          # MJHQ prompt查询工具
│   └── checkprompt_dci.py      # sDCI prompt查询工具
├── deepcompressor/             # 核心代码
│   ├── app/                    # 量化流程及配置
│   ├── backend/                # 后端部署适配器Nunchaku
│   ├── calib/                  # 搜索式/平滑/低秩校准
│   ├── csrc/                   # 量化CUDA内核
│   ├── data/                   # 量化数据结构
│   ├── dataset/                # 数据集处理
│   ├── nn/                     # 模型结构 补丁
│   ├── quantizer/              # 核心量化器实现/量化内核与配置
│   └── utils/                  # 通用组件
├── environment.yml             # 环境要求
└── pyproject.toml              # 项目依赖

```



## 项目简介

DeepCompressor 是一个面向大语言模型和扩散模型的模型压缩工具箱，支持多种量化算法：

- **大语言模型量化**：AWQ、GPTQ、SmoothQuant、QoQ (W4A8KV4)
- **扩散模型量化**：SVDQuant (W4A4)

## SVDQuant 算法

SVDQuant 是一种针对扩散模型的4位量化方法，通过低秩分支吸收异常值来实现高效的权重和激活量化：

1. **异常值转移**：将激活中的异常值转移到权重中
2. **低秩分支吸收**：使用SVD分解创建高精度低秩分支来吸收权重中的异常值
3. **Nunchaku推理引擎**：融合低秩分支和低位分支的计算核心，消除额外开销


## 实验环境

| 配置项 | 规格 |
|--------|------|
| GPU | NVIDIA A100 (1块) |
| GPU显存 | 80GB |
| CPU核心数 | 8核 |
| 系统内存 | 250GB |

## 复现实验

### 测试模型

| 模型 | 原始精度 | 推理步数 | Guidance Scale |
|------|----------|----------|----------------|
| FLUX.1-schnell | BFloat16 | 4 | 0 |
| PixArt-Sigma | Float16 | 20 | 4.5 |

### 评估数据集

- **MJHQ-30K**：高质量图像数据集，用于评估生成图像质量
- **sDCI**：用于评估生成图像的多样性和一致性

### 实验结果摘要

#### FLUX.1-schnell (MJHQ-1024)

| 精度/方法 | FID↓ | ImageReward↑ | LPIPS↓ | PSNR↑ |
|-----------|------|--------------|--------|-------|
| BF16 (Baseline) | 59.30 | 0.936 | -- | -- |
| INT4 SVDQuant | 58.63 | 0.924 | 0.291 | 17.50 |
| NVFP4 SVDQuant | 59.06 | **0.947** | **0.247** | **18.37** |

#### PixArt-Sigma (MJHQ-1024)

| 精度/方法 | FID↓ | ImageReward↑ | LPIPS↓ | PSNR↑ |
|-----------|------|--------------|--------|-------|
| FP16 (Baseline) | 56.76 | 0.929 | -- | -- |
| INT4 SVDQuant | 60.53 | 0.822 | 0.354 | 16.81 |

### 图片生成示例
#### 【Baseline】flux schnell / bf16 /MJHQ 
|![flux1_schnell_bf16_MJHQ](./examples/diffusion/baselines/torch.bfloat16/flux.1-schnell/fmeuler4-g0/samples/MJHQ/MJHQ-1024/0ab0643ff90b2617a724de13d919b45aecfcae66.png)|![flux1_schnell_bf16_MJHQ](./examples/diffusion/baselines/torch.bfloat16/flux.1-schnell/fmeuler4-g0/samples/MJHQ/MJHQ-1024/0aeafddbc82c62e2fe41346cccf48aba9ea38b84.png)|![flux1_schnell_bf16_MJHQ](./examples/diffusion/baselines/torch.bfloat16/flux.1-schnell/fmeuler4-g0/samples/MJHQ/MJHQ-1024/0afd5338eb2344c834bf41cc5a11352129d099d1.png)|![flux1_schnell_bf16_MJHQ](./examples/diffusion/baselines/torch.bfloat16/flux.1-schnell/fmeuler4-g0/samples/MJHQ/MJHQ-1024/0b8b917ab1c64395a6eb6bd58864671ad6a2b42b.png)
| :---: | :---: | :---: | :---: |
#### 【SVDQuant】flux schnell / int4 /MJHQ
|![flux1_schnell_int4_MJHQ](./outputs_example/flux1_schnell/int4/MJHQ/0ab0643ff90b2617a724de13d919b45aecfcae66.png)|![flux1_schnell_int4_MJHQ](./outputs_example/flux1_schnell/int4/MJHQ/0aeafddbc82c62e2fe41346cccf48aba9ea38b84.png)|![flux1_schnell_int4_MJHQ](./outputs_example/flux1_schnell/int4/MJHQ/0afd5338eb2344c834bf41cc5a11352129d099d1.png)|![flux1_schnell_int4_MJHQ](./outputs_example/flux1_schnell/int4/MJHQ/0b8b917ab1c64395a6eb6bd58864671ad6a2b42b.png)
| :---: | :---: | :---: | :---: |
#### 【SVDQuant】flux schnell / nvfp4 /MJHQ
|![flux1_schnell_nvfp4_MJHQ](./outputs_example/flux1_schnell/nvfp4/MJHQ/0ab0643ff90b2617a724de13d919b45aecfcae66.png)|![flux1_schnell_nvfp4_MJHQ](./outputs_example/flux1_schnell/nvfp4/MJHQ/0aeafddbc82c62e2fe41346cccf48aba9ea38b84.png)|![flux1_schnell_nvfp4_MJHQ](./outputs_example/flux1_schnell/nvfp4/MJHQ/0afd5338eb2344c834bf41cc5a11352129d099d1.png)|![flux1_schnell_nvfp4_MJHQ](./outputs_example/flux1_schnell/nvfp4/MJHQ/0b8b917ab1c64395a6eb6bd58864671ad6a2b42b.png)
| :---: | :---: | :---: | :---: |


## 配置及运行环境

### 1. 安装

```bash
cd deepcompressor
conda env create -f environment.yml
poetry install
```

### 2. 运行Baseline

```bash
cd examples/diffusion
python -m deepcompressor.app.diffusion.ptq configs/model/flux.1-schnell.yaml --output-dirname reference
```

### 3. 运行SVDQuant量化

```bash
python -m deepcompressor.app.diffusion.ptq \
    configs/model/flux.1-schnell.yaml configs/svdquant/int4.yaml \
    --eval-benchmarks MJHQ --eval-num-samples 1024
```
## 实验细节

[复现实验详细流程README](./examples/diffusion/README.md)

## 结果文件说明

每个实验目录`examples/diffusion/runs/`包含：

| 文件 | 描述 |
|------|------|
| `config-*.yaml` | 实验配置 |
| `results-*.json` | 评估指标结果 |
| `run-*.log` | 运行日志 |
| `samples/MJHQ/MJHQ-1024/` | 生成的图像样本 |
| `samples/DCI/sDCI-1024/` | DCI数据集生成样本 |

## 辅助工具

### 查询MJHQ数据集的Prompt

```bash
python docs/checkprompt.py
```

### 查询sDCI数据集的Prompt

```bash
python docs/checkprompt_dci.py
```

## 参考文献

1. Li, M., Lin, Y., Zhang, Z., et al. (2024). **SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models**. *arXiv preprint arXiv:2411.05007*.

2. Lin, Y., Tang, H., Yang, S., et al. (2024). **QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving**. *arXiv preprint arXiv:2405.04532*.

## 相关链接

- [DeepCompressor GitHub](https://github.com/mit-han-lab/deepcompressor)
- [Nunchaku 推理引擎](https://github.com/mit-han-lab/nunchaku)
- [SVDQuant 项目主页](https://hanlab.mit.edu/projects/svdquant)
- [SVDQuant 论文](http://arxiv.org/abs/2411.05007)

## 致谢

感谢 MIT HAN Lab 开源 DeepCompressor 项目。

