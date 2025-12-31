# -*- coding: utf-8 -*-
"""量化零点域模块。

本模块定义了量化过程中零点(zero-point)的计算域。
零点是非对称量化中的关键参数，用于将浮点数映射到整数范围。

量化公式:
    - 对称量化: q = round(x / scale)
    - 非对称量化: q = round(x / scale) + zero_point

零点域决定了零点在量化公式中的应用位置，这会影响：
    1. 量化精度
    2. 计算效率
    3. 硬件兼容性

主要组件:
    - ZeroPointDomain: 零点域枚举，定义零点的应用方式
"""

import enum

__all__ = ["ZeroPointDomain"]


class ZeroPointDomain(enum.Enum):
    """零点域枚举类。
    
    定义零点在量化/反量化过程中的应用位置。不同的零点域会影响
    量化的数值精度和计算效率。
    
    Attributes:
        PreScale: 预缩放零点域
            量化公式: q = round(x / s + z)
            反量化公式: x = (q - z) * s
            特点: 零点在缩放之前应用，适用于某些硬件加速器
            
        PostScale: 后缩放零点域
            量化公式: q = round((x + z) / s)
            反量化公式: x = q * s - z
            特点: 零点在缩放之后应用，计算上可能更高效
    
    Note:
        选择哪种零点域通常取决于目标硬件的支持情况和性能特性。
        例如，某些NPU/TPU可能对特定的零点域有更好的硬件支持。
    
    Example:
        >>> domain = ZeroPointDomain.PreScale
        >>> if domain == ZeroPointDomain.PreScale:
        ...     # 使用预缩放零点进行量化
        ...     q = torch.round(x / scale + zero_point)
    """

    PreScale = enum.auto()   # 预缩放零点域: q = round(x / s + z)
    PostScale = enum.auto()  # 后缩放零点域: q = round((x + z) / s)
