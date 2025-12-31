# -*- coding: utf-8 -*-
"""简单量化函数模块。

本模块提供了基础的量化函数，用于将浮点张量量化到指定的数据类型。
支持整数量化、浮点量化和指数量化三种模式。

量化模式:
    1. 整数量化: 使用四舍五入将浮点数映射到整数
    2. 浮点量化: 使用码本查找最近的可表示浮点值
    3. 指数量化: 在对数空间进行量化，适用于指数格式

主要函数:
    - simple_quantize: 简单量化函数，根据数据类型自动选择量化模式
"""

import torch

from ...data.dtype import QuantDataType
from ...data.range import LogQuantRange, QuantRange
from .ste import ste

__all__ = ["simple_quantize"]


def simple_quantize(
    tensor: torch.Tensor,
    *,
    quant_dtype: torch.dtype | QuantDataType,
    has_zero_point: bool,
    quant_range: QuantRange | None = None,
    round_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """简单量化函数。
    
    将浮点张量量化到指定的数据类型。支持三种量化模式：
    - torch.dtype: 直接类型转换
    - 整数QuantDataType: 四舍五入量化
    - 浮点QuantDataType: 码本查找量化
    - 指数QuantDataType: 对数空间量化

    Args:
        tensor (`torch.Tensor`):
            要量化的浮点张量。
        quant_dtype (`torch.dtype` or `QuantDataType`):
            目标量化数据类型。
        has_zero_point (`bool`):
            是否使用零点（非对称量化）。
            影响整数量化的范围计算。
        quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
            量化范��约束。如果指定，量化后的值会被裁剪到此范围。
        round_delta (`torch.Tensor` or `None`, *optional*, defaults to `None`):
            舍入增量。用于GPTQ等算法中的自适应舍入。
            如果指定，使用floor(x) + delta代替round(x)。

    Returns:
        `torch.Tensor`:
            量化后的张量。
            
    Example:
        >>> # 整数量化
        >>> x = torch.randn(4, 4)
        >>> q = simple_quantize(x, quant_dtype=QDType.sint4, has_zero_point=False)
        >>> 
        >>> # 浮点量化
        >>> q = simple_quantize(x, quant_dtype=QDType.sfp4_e2m1, has_zero_point=False)
    
    Note:
        - 对于整数量化，round_delta可用于实现自适应舍入
        - 对于浮点和指数量化，round_delta不支持
        - 函数会自动处理梯度传播（使用STE）
    """
    requires_grad = tensor.requires_grad
    
    if isinstance(quant_dtype, torch.dtype):
        # PyTorch原生数据类型：直接类型转换
        dtype = tensor.dtype
        tensor = tensor.to(dtype=quant_dtype).to(dtype=dtype)
        if round_delta is not None:
            tensor = tensor.add_(round_delta)
        if quant_range is not None and quant_range.is_set():
            tensor = torch.clamp(tensor, min=quant_range.min, max=quant_range.max)
        return tensor
    
    elif isinstance(quant_dtype, QuantDataType):
        if quant_dtype.is_exponent:
            # 指数量化：在对数空间进行量化
            # 适用于纯指数格式（无尾数位）
            assert round_delta is None, "round_delta is not supported for exponential quantization"
            quant_range = LogQuantRange.construct(quant_dtype, quant_range)
            # 转换到对数空间，向下取整，然后裁剪到有效指数范围
            tensor = ste(tensor.log2(), torch.floor) if requires_grad else tensor.log2_().floor_()
            return tensor.clamp_(min=quant_range.min, max=quant_range.max).exp2_()
        
        elif quant_dtype.is_float_point:
            # 浮点量化：使用码本查找最近的可表示值
            assert round_delta is None, "round_delta is not supported for float quantization"
            # 先裁剪到数据类型的有效范围
            tensor = torch.clamp(tensor, min=quant_dtype.min_value, max=quant_dtype.max_value)
            # 使用码本四舍五入到最近的可表示值
            tensor = ste(tensor, quant_dtype.round)
            if quant_range is not None and quant_range.is_set():
                tensor = tensor.clamp_(min=quant_range.min, max=quant_range.max)
            return tensor
        
        else:
            # 整数量化：标准的四舍五入量化
            quant_range = QuantRange.construct(quant_dtype, has_zero_point=has_zero_point, quant_range=quant_range)
            if round_delta is None:
                # 标准四舍五入
                tensor = ste(tensor, torch.round) if requires_grad else tensor.round_()
            else:
                # 自适应舍入：floor(x) + delta
                # 用于GPTQ等算法，delta是学习得到的舍入决策
                tensor = ste(tensor, torch.floor) if requires_grad else tensor.floor_()
                tensor = tensor.add_(round_delta)
            # 裁剪到量化范围
            return tensor.clamp_(min=quant_range.min, max=quant_range.max)
    
    else:
        raise TypeError(
            f"quant_dtype must be either torch.dtype or QuantDataType, got {quant_dtype} ({type(quant_dtype)})"
        )
