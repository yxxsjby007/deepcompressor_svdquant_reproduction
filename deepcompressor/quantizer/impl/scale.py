# -*- coding: utf-8 -*-
"""量化缩放因子计算模块。

本模块实现了量化过程中缩放因子(scale)的计算和量化。
缩放因子是量化的核心参数，决定了浮点数到量化值的映射关系。

缩放因子的作用:
    量化: q = round(x / scale)
    反量化: x' = q * scale

在分组量化中，缩放因子本身也可以被量化以节省存储空间。
这形成了层次化的缩放因子结构。

主要组件:
    - quantize_scale: 量化缩放因子张量
    - QuantScaleInfo: 缩放因子信息类，管理量化参数
"""

import math
import typing as tp
from dataclasses import dataclass, field

import torch

from ...data.dtype import QuantDataType
from ...data.range import DynamicRange, QuantRange, RangeBound
from ...data.scale import QuantScale
from ...data.utils import ScaleUtils
from ...data.zero import ZeroPointDomain
from .simple import simple_quantize

__all__ = ["quantize_scale", "QuantScaleInfo"]


def quantize_scale(
    s: torch.Tensor,
    /,
    *,
    quant_dtypes: tp.Sequence[QuantDataType],
    quant_spans: tp.Sequence[float],
    view_shapes: tp.Sequence[torch.Size],
) -> QuantScale:
    """量化缩放因子张量。
    
    将缩放因子张量按层次结构进行量化。每一层使用不同的量化参数，
    形成从粗粒度到细粒度的缩放因子层次。

    Args:
        s (`torch.Tensor`):
            原始缩放因子张量。
        quant_dtypes (`Sequence[QuantDataType]`):
            每层缩放因子的量化数据类型。
        quant_spans (`Sequence[float]`):
            每层缩放因子的量化跨度（用于计算缩放因子的缩放因子）。
        view_shapes (`Sequence[torch.Size]`):
            每层缩放因子的视图形状，格式为 (#g0, gs0, #g1, gs1, ...)。

    Returns:
        `QuantScale`:
            量化后的层次化缩放因子对象。
            
    Example:
        >>> # 两级缩放因子量化
        >>> s = torch.randn(4, 128).abs()  # 原始缩放因子
        >>> qs = quantize_scale(
        ...     s,
        ...     quant_dtypes=[QDType.sfp8_e4m3, QDType.fp16],
        ...     quant_spans=[1.0, 1.0],
        ...     view_shapes=[torch.Size([4, 32, 1, 4]), torch.Size([4, 128])]
        ... )
    """
    scale = QuantScale()
    s = s.abs()  # 缩放因子始终为正
    
    # 逐层量化缩放因子（除最后一层外）
    for view_shape, quant_dtype, quant_span in zip(view_shapes[:-1], quant_dtypes[:-1], quant_spans[:-1], strict=True):
        s = s.view(view_shape)  # 重塑为 (#g0, rs0, #g1, rs1, #g2, rs2, ...)
        # 计算当前层的动态范围（组内最大值）
        ss = s.amax(dim=list(range(1, len(view_shape), 2)), keepdim=True)  # 即 s_dynamic_span
        # 量化当前层的缩放因子
        # s_scale = s_dynamic_span / s_quant_span
        ss = simple_quantize(
            ss / quant_span, has_zero_point=False, quant_dtype=quant_dtype
        )
        # 归一化：除以当前层的缩放因子
        s = s / ss
        scale.append(ss)
    
    # 处理最后一层
    view_shape = view_shapes[-1]
    s = s.view(view_shape)
    if any(v != 1 for v in view_shape[1::2]):
        # 如果最后一层有分组，计算组内最大值
        ss = s.amax(dim=list(range(1, len(view_shape), 2)), keepdim=True)
        ss = simple_quantize(ss / quant_spans[-1], has_zero_point=False, quant_dtype=quant_dtypes[-1])
    else:
        # 如果最后一层无分组，直接量化
        assert quant_spans[-1] == 1, "The last quant span must be 1."
        ss = simple_quantize(s, has_zero_point=False, quant_dtype=quant_dtypes[-1])
    scale.append(ss)
    
    # 移除零值缩放因子（避免除零错误）
    scale.remove_zero()
    return scale


@dataclass
class QuantScaleInfo:
    """量化缩放因子信息类。
    
    管理量化过程中缩放因子的计算参数，包括：
    - 张量的量化参数（数据类型、零点域、范围等）
    - 缩放因子的量化参数（数据类型、视图形状等）
    - 线性缩放和指数缩放的分离处理
    
    Attributes:
        tensor_view_shape: 张量的视图形状
        tensor_quant_dtype: 张量的量化数据类型
        tensor_zero_domain: 张量的零点域
        tensor_quant_range: 张量的量化范围
        tensor_range_bound: 张量的范围边界
        default_quant_dtype: 默认的量化数据类型
        scale_view_shapes: 缩放因子的视图形状列表
        scale_quant_dtypes: 缩放因子的量化数据类型列表
    """
    
    # region 张量信息
    tensor_view_shape: torch.Size           # 张量视图形状
    tensor_quant_dtype: torch.dtype | QuantDataType  # 张量量化数据类型
    tensor_zero_domain: ZeroPointDomain | None       # 零点域
    tensor_quant_range: QuantRange          # 量化范围
    tensor_range_bound: RangeBound | None   # 范围边界
    # endregion
    
    default_quant_dtype: torch.dtype | QuantDataType  # 默认量化数据类型
    scale_view_shapes: list[torch.Size]     # 缩放因子视图形状
    scale_quant_dtypes: list[torch.dtype | QuantDataType]  # 缩放因子数据类型
    
    # 以下字段在__post_init__中初始化
    exponent_scale_level: int = field(init=False)  # 指数缩放的起始层级
    zero_quant_dtype: torch.dtype | QuantDataType = field(init=False)  # 零点的量化类型
    
    # region 线性缩放信息
    linear_tensor_quant_span: float = field(init=False)  # 线性量化跨度
    linear_scale_quant_dtypes: list[torch.dtype | QuantDataType] = field(init=False)
    linear_scale_view_shapes: list[torch.Size] = field(init=False)
    linear_scale_quant_spans: list[float] = field(init=False)
    # endregion
    
    # region 指数缩放信息
    exponent_tensor_quant_span: float = field(init=False)  # 指数量化跨度
    exponent_scale_quant_dtypes: list[torch.dtype | QuantDataType] = field(init=False)
    exponent_scale_view_shapes: list[torch.Size] = field(init=False)
    exponent_scale_quant_spans: list[float] = field(init=False)
    # endregion

    @property
    def has_zero_point(self) -> bool:
        """是否使用零点（非对称量化）。"""
        return self.tensor_zero_domain is not None

    def __post_init__(self):
        """初始化后处理：计算各种量化参数。"""
        if isinstance(self.tensor_quant_dtype, torch.dtype):
            raise NotImplementedError("torch.dtype is not supported yet.")
        
        # 构建最终的量化范围
        self.tensor_quant_range = QuantRange.construct(
            self.tensor_quant_dtype, has_zero_point=self.has_zero_point, quant_range=self.tensor_quant_range
        )
        
        # 推断缩放因子的数据类型
        self.scale_quant_dtypes = ScaleUtils.infer_scale_dtypes(self.scale_quant_dtypes, self.default_quant_dtype)
        
        # 确定指数缩放的起始层级
        self.exponent_scale_level = ScaleUtils.infer_exponent_scale_level(self.scale_quant_dtypes)
        
        # 确定零点的量化数据类型
        if self.has_zero_point:
            if self.tensor_zero_domain == ZeroPointDomain.PreScale:
                # 预缩放零点：使用与张量相同的数据类型
                self.zero_quant_dtype = self.tensor_quant_dtype
            elif self.tensor_zero_domain == ZeroPointDomain.PostScale:
                # 后缩放零点：使用缩放因子的数据类型
                self.zero_quant_dtype = self.scale_quant_dtypes[-1]
                if isinstance(self.zero_quant_dtype, QuantDataType) and self.zero_quant_dtype.is_exponent:
                    self.zero_quant_dtype = self.default_quant_dtype
            else:
                raise ValueError(f"Unsupported zero point domain: {self.tensor_zero_domain}")
            # 非对称量化的量化跨度
            self.linear_tensor_quant_span = self.tensor_quant_range.max - self.tensor_quant_range.min
            self.exponent_tensor_quant_span = 2 ** int(
                math.log2(self.tensor_quant_range.max) + int(self.tensor_quant_dtype.signed)
            )
        else:
            self.zero_quant_dtype = None
            # 对称量化的量化跨度
            self.linear_tensor_quant_span = self.tensor_quant_range.max
            self.exponent_tensor_quant_span = 2 ** int(math.log2(self.tensor_quant_range.max))
        
        # 分离线性缩放和指数缩放
        if self.exponent_scale_level >= 0 and self.exponent_scale_level < len(self.scale_quant_dtypes):
            lin_s_dtypes = self.scale_quant_dtypes[: self.exponent_scale_level]
            exp_s_dtypes = self.scale_quant_dtypes[self.exponent_scale_level :]
            lin_s_view_shapes = self.scale_view_shapes[: self.exponent_scale_level]
            exp_s_view_shapes = self.scale_view_shapes[self.exponent_scale_level :]
            exp_s_spans = ScaleUtils.infer_scale_quant_spans(exp_s_dtypes)
            lin_s_spans = ScaleUtils.infer_scale_quant_spans(lin_s_dtypes, base=exp_s_spans[-1]) if lin_s_dtypes else []
        else:
            lin_s_dtypes, exp_s_dtypes = self.scale_quant_dtypes, []
            lin_s_view_shapes, exp_s_view_shapes = self.scale_view_shapes, []
            lin_s_spans, exp_s_spans = ScaleUtils.infer_scale_quant_spans(lin_s_dtypes), []
        
        self.linear_scale_quant_dtypes = lin_s_dtypes
        self.linear_scale_view_shapes = lin_s_view_shapes
        self.linear_scale_quant_spans = lin_s_spans
        self.exponent_scale_quant_dtypes = exp_s_dtypes
        self.exponent_scale_view_shapes = exp_s_view_shapes
        self.exponent_scale_quant_spans = exp_s_spans

    def quantize(
        self,
        *,
        # 基于缩放因子的量化参数
        scale: torch.Tensor | None = None,
        zero: torch.Tensor | None = None,
        # 基于范围的量化参数
        tensor: torch.Tensor | None = None,
        dynamic_range: DynamicRange | None = None,
    ) -> tuple[QuantScale, torch.Tensor]:
        """计算量化的缩放因子和零点。
        
        支持两种模式：
        1. 基于缩放因子：直接使用提供的scale和zero
        2. 基于范围：从tensor测量动态范围，计算scale和zero

        Args:
            scale (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                预设的缩放因子张量。如果提供，使用基于缩放因子的模式。
            zero (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                预设的零点张量。
            tensor (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                待量化的张量。用于基于范围的模式。
            dynamic_range (`DynamicRange` or `None`, *optional*, defaults to `None`):
                动态范围配置。

        Returns:
            `tuple[QuantScale, torch.Tensor]`:
                (量化后的缩放因子, 量化后的零点)
        """
        # region 步骤1: 获取动态跨度（基于范围）或缩放因子张量
        if scale is None:
            range_based = True
            assert isinstance(tensor, torch.Tensor), "View tensor must be a tensor."
            dynamic_range = dynamic_range or DynamicRange()
            # 从张量测量动态范围
            dynamic_range = dynamic_range.measure(
                tensor.view(self.tensor_view_shape),
                zero_domain=self.tensor_zero_domain,
                is_float_point=self.tensor_quant_dtype.is_float_point,
            )
            dynamic_range = dynamic_range.intersect(self.tensor_range_bound)
            # 计算动态跨度
            dynamic_span = (dynamic_range.max - dynamic_range.min) if self.has_zero_point else dynamic_range.max
        else:
            range_based = False
            scale = scale.view(self.scale_view_shapes[-1])
            assert isinstance(scale, torch.Tensor), "Scale must be a tensor."
        # endregion
        
        # region 步骤2: 计算缩放因子
        if self.linear_scale_quant_dtypes:
            if range_based:
                # 基于范围：scale = dynamic_span / quant_span
                linear_scale = dynamic_span / self.linear_tensor_quant_span
            elif self.exponent_scale_quant_dtypes:
                # 有指数缩放：调整线性缩放
                linear_scale = scale.mul(self.exponent_tensor_quant_span).div(self.linear_tensor_quant_span)
            else:
                linear_scale = scale
            # 量化线性缩放因子
            lin_s = quantize_scale(
                linear_scale,
                quant_dtypes=self.linear_scale_quant_dtypes,
                quant_spans=self.linear_scale_quant_spans,
                view_shapes=self.linear_scale_view_shapes,
            )
            assert lin_s.data is not None, "Linear scale tensor is None."
            assert not lin_s.data.isnan().any(), "Linear scale tensor contains NaN."
            assert not lin_s.data.isinf().any(), "Linear scale tensor contains Inf."
        else:
            lin_s = QuantScale()
        
        if self.exponent_scale_quant_dtypes:
            if range_based:
                exp_scale = dynamic_span / self.exponent_tensor_quant_span
            else:
                exp_scale = scale
            if lin_s.data is not None:
                # 调整指数缩放以考虑线性缩放
                lin_s.data = lin_s.data.expand(self.linear_scale_view_shapes[-1]).reshape(self.scale_view_shapes[-1])
                exp_scale = exp_scale / lin_s.data
            # 量化指数缩放因子
            exp_s = quantize_scale(
                exp_scale,
                quant_dtypes=self.exponent_scale_quant_dtypes,
                quant_spans=self.exponent_scale_quant_spans,
                view_shapes=self.exponent_scale_view_shapes,
            )
            assert exp_s.data is not None, "Exponential scale tensor is None."
            assert not exp_s.data.isnan().any(), "Exponential scale tensor contains NaN."
            assert not exp_s.data.isinf().any(), "Exponential scale tensor contains Inf."
            # 合并线性和指数缩放
            s = exp_s if lin_s.data is None else lin_s.extend(exp_s)
        else:
            s = lin_s
        
        assert s.data is not None, "Scale tensor is None."
        assert not s.data.isnan().any(), "Scale tensor contains NaN."
        assert not s.data.isinf().any(), "Scale tensor contains Inf."
        # endregion
        
        # region 步骤3: 计算零点
        if self.has_zero_point:
            if range_based:
                # 根据零点域计算零点
                if self.tensor_zero_domain == ZeroPointDomain.PreScale:
                    # 预缩放零点: z = qmin - vmin / s
                    zero = self.tensor_quant_range.min - dynamic_range.min / s.data
                else:
                    # 后缩放零点: z = qmin * s - vmin
                    zero = self.tensor_quant_range.min * s.data - dynamic_range.min
            assert isinstance(zero, torch.Tensor), "Zero point must be a tensor."
            # 量化零点
            z = simple_quantize(zero, has_zero_point=True, quant_dtype=self.zero_quant_dtype)
        else:
            z = torch.tensor(0, dtype=s.data.dtype, device=s.data.device)
        
        assert not z.isnan().any(), "Zero point tensor contains NaN."
        assert not z.isinf().any(), "Zero point tensor contains Inf."
        # endregion
        
        return s, z
