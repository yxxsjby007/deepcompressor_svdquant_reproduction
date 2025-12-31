# -*- coding: utf-8 -*-
"""RTN (Round-To-Nearest) 量化核心模块。

本模块实现了最基础的四舍五入量化算法。RTN是最简单的量化方法，
直接将浮点数四舍五入到最近的量化值。

RTN量化流程:
    1. 应用零点（如果是后缩放零点域）
    2. 除以缩放因子
    3. 应用零点（如果是预缩放零点域）
    4. 四舍五入到最近整数
    5. 裁剪到量化范围

优点:
    - 实现简单，计算快速
    - 不需要校准数据
    - 适用于大多数场景

缺点:
    - 对于低位宽量化（如4位），精度损失较大
    - 不考虑权重的重要性差异

主要组件:
    - QuantRtnKernel: RTN量化核心类
    - rtn_quantize: RTN量化函数
"""

import torch

from ...data.dtype import QuantDataType
from ...data.range import QuantRange
from ...data.zero import ZeroPointDomain
from ..config.kernel import BaseQuantKernel
from ..impl.simple import simple_quantize

__all__ = ["QuantRtnKernel", "rtn_quantize"]


class QuantRtnKernel(BaseQuantKernel):
    """RTN (Round-To-Nearest) 量化核心类。
    
    实现了基于四舍五入的简单量化算法。这是默认的量化核心，
    当没有指定其他量化核心时使用。
    
    Example:
        >>> kernel = QuantRtnKernel()
        >>> qtensor = kernel.quantize(
        ...     tensor=weight,
        ...     view_shape=view_shape,
        ...     quant_dtype=QDType.sint4,
        ...     zero_domain=None,
        ...     scale=scale,
        ...     zero=zero,
        ... )
    """

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        view_shape: torch.Size,
        quant_dtype: QuantDataType,
        zero_domain: ZeroPointDomain | None,
        scale: torch.Tensor,
        zero: torch.Tensor,
        quant_range: QuantRange | None = None,
        round_delta: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """执行RTN量化。

        Args:
            tensor (`torch.Tensor`):
                要量化的张量。
            view_shape (`torch.Size`):
                量化时的视图形状，格式为 (#g0, gs0, #g1, gs1, ...)。
            quant_dtype (`QuantDataType`):
                量化数据类型。
            zero_domain (`ZeroPointDomain` or `None`):
                零点域。None表示对称量化。
            scale (`torch.Tensor`):
                缩放因子张量。
            zero (`torch.Tensor`):
                零点张量。
            quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                量化范围约束。
            round_delta (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                舍入增量，用于自适应舍入。
            **kwargs: 其他关键字参数（被忽略）。

        Returns:
            `torch.Tensor`:
                量化后的张量，形状为 ``view_shape``。
        """
        return rtn_quantize(
            tensor,
            view_shape=view_shape,
            quant_dtype=quant_dtype,
            zero_domain=zero_domain,
            scale=scale,
            zero=zero,
            quant_range=quant_range,
            round_delta=round_delta,
        )


def rtn_quantize(
    tensor: torch.Tensor,
    *,
    view_shape: torch.Size,
    quant_dtype: QuantDataType,
    zero_domain: ZeroPointDomain | None,
    scale: torch.Tensor,
    zero: torch.Tensor,
    quant_range: QuantRange | None = None,
    round_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """使用RTN算法量化张量。
    
    RTN (Round-To-Nearest) 是最简单的量化方法，直接将缩放后的值
    四舍五入到最近的整数���
    
    量化公式（对称量化）:
        q = round(x / scale)
        
    量化公式（非对称量化，预缩放零点）:
        q = round(x / scale + zero)
        
    量化公式（非对称量化，后缩放零点）:
        q = round((x + zero) / scale)

    Args:
        tensor (`torch.Tensor`):
            要量化的张量。
        view_shape (`torch.Size`):
            量化时的视图形状。
        quant_dtype (`QuantDataType`):
            量化数据类型。
        zero_domain (`ZeroPointDomain` or `None`):
            零点域。
        scale (`torch.Tensor`):
            缩放因子张量。
        zero (`torch.Tensor`):
            零点张量。
        quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
            量化范围约束。
        round_delta (`torch.Tensor` or `None`, *optional*, defaults to `None`):
            舍入增量。如果提供，使用 floor(x) + delta 代替 round(x)。

    Returns:
        `torch.Tensor`:
            量化后的张量，形状为 ``view_shape``。
            
    Note:
        返回的是量化后的整数值（但数据类型仍为浮点），
        需要后续的反量化步骤才能得到近似的原始值。
    """
    # 重塑张量为分组视图
    qtensor = tensor.view(view_shape)
    round_delta = round_delta.view(view_shape) if round_delta is not None else None
    
    # 步骤1: 应用后缩放零点（如果适用）
    if zero_domain == ZeroPointDomain.PostScale:
        qtensor = qtensor.add_(zero)
    
    # 步骤2: 除以缩放因子
    qtensor = qtensor.div(scale)
    
    # 步骤3: 应用预缩放零点（如果适用）
    if zero_domain == ZeroPointDomain.PreScale:
        qtensor = qtensor.add_(zero)
    
    # 步骤4: 四舍五入并裁剪到量化范围
    qtensor = simple_quantize(
        qtensor,
        quant_dtype=quant_dtype,
        has_zero_point=zero_domain is not None,
        quant_range=quant_range,
        round_delta=round_delta,
    )
    
    return qtensor
