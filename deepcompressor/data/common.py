# -*- coding: utf-8 -*-
"""通用量化数据模块。

本模块定义了量化过程中使用的通用数据类型和枚举。
在深度学习模型压缩中，我们需要区分不同类型的张量（权重、输入、输出），
因为它们可能需要不同的量化策略和处理方式。

主要组件:
    - TensorType: 张量类型枚举，用于标识张量在模型中的角色
"""

import enum

__all__ = ["TensorType"]


class TensorType(enum.Enum):
    """张量类型枚举类。
    
    在模型量化过程中，不同类型的张量通常需要不同的处理策略：
    - 权重(Weights): 通常是静态的，可以离线量化，支持更激进的压缩
    - 输入(Inputs): 动态数据，需要在线量化，通常需要保持较高精度
    - 输出(Outputs): 层的输出激活值，可能需要特殊的量化范围处理
    
    Attributes:
        Weights: 模型权重张量，如线性层���权重矩阵、卷积核等
        Inputs: 输入激活张量，即层的输入数据
        Outputs: 输出激活张量，即层的输出数据
    
    Example:
        >>> tensor_type = TensorType.Weights
        >>> if tensor_type == TensorType.Weights:
        ...     # 应用权重专用的量化策略
        ...     pass
    """

    Weights = enum.auto()   # 权重张量
    Inputs = enum.auto()    # 输入激活张量
    Outputs = enum.auto()   # 输出激活张量
