# -*- coding: utf-8 -*-
"""量化张量模块。

本模块定义了量化张量的数据结构，用于存储量化过程中的各种信息。
量化张量包含原始数据、量化后的数据、缩放因子和零点等信息。

量化过程:
    1. 计算缩放因子(scale)和零点(zero_point)
    2. 量化: q = round(x / scale) + zero_point
    3. 反量化: x' = (q - zero_point) * scale

主要组件:
    - QuantTensor: 量化张量类，封装量化相关的所有数据
"""

import torch

from .scale import QuantScale

__all__ = ["QuantTensor"]


class QuantTensor:
    """量化张量类。
    
    封装了量化过程中的所有相关数据，包括：
    - 反量化后的张量（用于计算）
    - 量化后的整数张量（用于存储和推理）
    - 缩放因子和零点（用于量化/反量化转换）
    
    Attributes:
        _dequantized: 反量化后的浮点张量
        _quantized: 量化后的整数张量
        scale: 量化缩放因子
        zero: 量化零点
        view_shape: 用于分组量化的视图形状
    
    Example:
        >>> # 创建一个量化张量
        >>> dequant_data = torch.randn(4, 4)
        >>> scale = QuantScale()
        >>> scale.append(torch.tensor([0.1]))
        >>> qtensor = QuantTensor(dequantized=dequant_data, scale=scale)
        >>> print(qtensor.data.shape)  # torch.Size([4, 4])
    """

    _dequantized: torch.Tensor | None   # 反量化后的浮点张量
    _quantized: torch.Tensor | None     # 量化后的整数张量
    scale: QuantScale | None            # 量化缩放因子
    zero: torch.Tensor | float | None   # 量化零点
    view_shape: torch.Size | None       # 分组量化的视图形状

    def __init__(
        self,
        dequantized: torch.Tensor | None = None,
        quantized: torch.Tensor | None = None,
        scale: QuantScale | None = None,
        zero: torch.Tensor | float | None = None,
        view_shape: torch.Size | None = None,
    ):
        """初始化量化张量。
        
        Args:
            dequantized: 反量化后的浮点张量。用于训练和精度评估。
            quantized: 量化后的整数张量。用于推理和存储。
            scale: 量化缩放因子。用于量化/反量化转换。
            zero: 量化零点。用于非对称量化。
            view_shape: 视图形状。用于分组量化时的张量重塑。
            
        Note:
            dequantized 和 quantized 至少需要提供一个。
        """
        assert (
            dequantized is not None or quantized is not None
        ), "Either the dequantized or quantized tensor must be provided."
        self.view_shape = view_shape
        self._dequantized = dequantized
        self._quantized = quantized
        self.scale = scale
        self.zero = zero

    @property
    def data(self) -> torch.Tensor | None:
        """获取反量化后的浮点张量。
        
        Returns:
            反量化后的张量，用于计算和训练。
        """
        return self._dequantized

    @property
    def qdata(self) -> torch.Tensor | None:
        """获取量化后的整数张量。
        
        Returns:
            量化后的张量，用于推理和存储。
        """
        return self._quantized
