# -*- coding: utf-8 -*-
"""量化器模块。

本模块提供了模型量化的核心实现，包括：
- 量化器类：执行实际的量化操作
- 量化配置：定义量化参数和策略
- 量化核心：实现具体的量化算法（RTN、GPTQ等）

支持的量化算法:
    - RTN (Round-To-Nearest): 最简单的四舍五入量化
    - GPTQ: 基于Hessian的逐层量化优化
    - AWQ: 激活感知的权重量化
    - SmoothQuant: 平���量化

主要导出:
    - Quantizer: 量化器类，提供完整的量化功能
"""

from .processor import Quantizer
