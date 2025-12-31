# -*- coding: utf-8 -*-
"""量化处理器模块。

本模块定义了Quantizer类，它是量化操作的主要入口点。
Quantizer封装了量化的所有配置和状态，提供了简洁的API进行张量量化。

主要功能:
    - 张量量化和反量化
    - 支持多种量化核心（RTN、GPTQ等）
    - 支持低秩分支补偿（用于SVDQuant）
    - 状态保存和加载

使用示例:
    >>> from deepcompressor.quantizer import Quantizer
    >>> from deepcompressor.quantizer.config import QuantizerConfig
    >>> 
    >>> # 创建量化器
    >>> config = QuantizerConfig(dtype="sint4")
    >>> quantizer = Quantizer(config=config)
    >>> 
    >>> # 量化张量
    >>> weight = torch.randn(128, 256)
    >>> qtensor = quantizer.quantize(weight)
    >>> quantized_weight = qtensor.data  # 反量化后的权重
"""

import typing as tp
from dataclasses import _MISSING_TYPE, MISSING, dataclass

import torch

from ..data.range import DynamicRange, QuantRange, RangeBound
from ..data.tensor import QuantTensor
from ..nn.patch.lowrank import LowRankBranch
from ..utils.common import tree_map
from ..utils.config import KeyEnableConfig
from ..utils.hooks import BaseInputPackager, BaseOutputPackager, BaseTensorProcessor
from .config.kernel import BaseKeyEnableQuantKernelConfig, BaseQuantKernel, BaseQuantKernelConfig
from .config.lowrank import QuantLowRankConfig
from .impl.base import QuantizerImpl
from .impl.info import QuantInfo

__all__ = ["Quantizer"]


@dataclass
class Quantizer(QuantizerImpl, BaseTensorProcessor):
    """量化器类。
    
    继承自QuantizerImpl和BaseTensorProcessor，提供完整的量化功能。
    支持配置化的量化参数、多种量化核心、低秩分支补偿等高级特性。

    Args:
        config (`BasicQuantizerConfig` or `None`):
            量化器配置，定义量化的数据类型、分组策略等。
        key (`str`, *optional*, defaults to `""`):
            量化器的键名，用于区分不同的量化器实例。
        kernel (`BaseKeyEnableQuantKernelConfig` or `BaseQuantKernelConfig` or `BaseQuantKernel` or `None`,
                *optional*, defaults to `None`):
            量化核心配置，定义具体的量化算法（如RTN、GPTQ）。
        channels_dim (`int` or `None`, *optional*, defaults to `None`):
            通道维度，用于确定量化的分组方向。
        scale (`torch.Tensor` or `Sequence[torch.Tensor]` or `None`, *optional*, defaults to `None`):
            预设的缩放因子张量。
        zero (`torch.Tensor` or `None`, *optional*, defaults to `None`):
            预设的零点张量。
        dynamic_range (`DynamicRange` or `Sequence[DynamicRange]` or `None`, *optional*, defaults to `None`):
            动态范围配置。
        range_bound (`RangeBound` or `None`, *optional*, defaults to `None`):
            范围边界约束。
        quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
            量化范围约束。
        default_dtype (`torch.dtype` or `None`, *optional*, defaults to `None`):
            缩放因子的默认数据类型。
        develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            量化计算时使用的数据类型（通常使用高精度）。
        low_rank (`QuantLowRankConfig` or `None`, *optional*, defaults to `None`):
            低秩分支配置，用于SVDQuant等算法。
        input_packager (`BaseInputPackager` or `None`, *optional*, defaults to `None`):
            输入打包器，用于处理复杂的输入格式。
        output_packager (`BaseOutputPackager` or `None`, *optional*, defaults to `None`):
            输出打包器，用于处理复杂的输出格式。
    
    Example:
        >>> # 基本使用
        >>> quantizer = Quantizer(config=config)
        >>> qtensor = quantizer.quantize(weight)
        >>> 
        >>> # 使用低秩补偿
        >>> qtensors, branches = quantizer.quantize_with_low_rank(weight)
    """

    # region 量化参数默认值
    kernel: BaseKeyEnableQuantKernelConfig | BaseQuantKernelConfig | BaseQuantKernel | None = None
    channels_dim: int | None = None                                          # 通道维度
    scale: torch.Tensor | tp.Sequence[torch.Tensor] | None = None           # 缩放因子
    zero: torch.Tensor | None = None                                         # 零点
    dynamic_range: DynamicRange | tp.Sequence[DynamicRange] | None = None   # 动态范围
    range_bound: RangeBound | None = None                                    # 范围边界
    quant_range: QuantRange | None = None                                    # 量化范围
    default_dtype: torch.dtype | None = None                                 # 默认数据类型
    develop_dtype: torch.dtype = torch.float32                               # 计算数据类型
    # endregion
    
    # region Hook相关属性
    low_rank: QuantLowRankConfig | None = None          # 低秩分支配置
    input_packager: BaseInputPackager | None = None     # 输入打包器
    output_packager: BaseOutputPackager | None = None   # 输出打包器
    # endregion

    def is_enabled_low_rank(self) -> bool:
        """检查低秩分支是否启用。
        
        Returns:
            如果低秩分支配置存在且启用，返回True。
        """
        if self.low_rank is None:
            return False
        if isinstance(self.low_rank, KeyEnableConfig):
            return self.low_rank.is_enabled_for(self.key)
        return self.low_rank.is_enabled()

    def get_input_packager(self) -> BaseInputPackager | None:
        """获取输入打包器。"""
        return self.input_packager

    def get_output_packager(self) -> BaseOutputPackager | None:
        """获取输出打包器。"""
        return self.output_packager

    def process(self, tensor: torch.Tensor) -> torch.Tensor:
        """处理张量（实现BaseTensorProcessor接口）。
        
        Args:
            tensor: 输入张量。
            
        Returns:
            量化并反量化后的张量。
        """
        return self.quantize(tensor).data

    def quantize(
        self,
        tensor: torch.Tensor,
        /,
        *,
        return_with_dequant: bool = True,
        return_with_quant: bool = False,
        kernel: (
            BaseKeyEnableQuantKernelConfig | BaseQuantKernelConfig | BaseQuantKernel | None | _MISSING_TYPE
        ) = MISSING,
        channels_dim: int | None | _MISSING_TYPE = MISSING,
        # scale-based quantization arguments
        scale: torch.Tensor | tp.Sequence[torch.Tensor] | None | _MISSING_TYPE = MISSING,
        zero: torch.Tensor | None | _MISSING_TYPE = MISSING,
        # range-based quantization arguments
        dynamic_range: DynamicRange | tp.Sequence[DynamicRange] | None | _MISSING_TYPE = MISSING,
        range_bound: RangeBound | None | _MISSING_TYPE = MISSING,
        # other arguments
        quant_range: QuantRange | None | _MISSING_TYPE = MISSING,
        default_dtype: torch.dtype | None | _MISSING_TYPE = MISSING,
        develop_dtype: torch.dtype | _MISSING_TYPE = MISSING,
        **kwargs,
    ) -> QuantTensor:
        """量化张量。
        
        将浮点张量量化为低精度表示，并可选地返回反量化后的张量。

        Args:
            tensor (`torch.Tensor`):
                要量化的浮点张量。
            return_with_dequant (`bool`, *optional*, defaults to `True`):
                是否返回反量化后的张量。
            return_with_quant (`bool`, *optional*, defaults to `False`):
                是否返回量化后的整数张量。
            kernel: 量化核心配置，定义具体的量化算法。
            channels_dim: 通道维度。
            scale: 预设的缩放因子。
            zero: 预设的零点。
            dynamic_range: 动态范围。
            range_bound: 范围边界。
            quant_range: 量化范围。
            default_dtype: 缩放因子的默认数据类型。
            develop_dtype: 计算时使用的数据类型。
            **kwargs: 传递给量化核心的其他参数。

        Returns:
            QuantTensor: 包含量化结果的对象。
            
        Note:
            参数值为MISSING时，使用实例的默认值。
        """
        # 使用实例默认值填充MISSING参数
        channels_dim = self.channels_dim if channels_dim is MISSING else channels_dim
        scale = self.scale if scale is MISSING else scale
        zero = self.zero if zero is MISSING else zero
        dynamic_range = self.dynamic_range if dynamic_range is MISSING else dynamic_range
        range_bound = self.range_bound if range_bound is MISSING else range_bound
        quant_range = self.quant_range if quant_range is MISSING else quant_range
        default_dtype = self.default_dtype if default_dtype is MISSING else default_dtype
        develop_dtype = self.develop_dtype if develop_dtype is MISSING else develop_dtype
        
        # 处理量化核心配置
        if kernel is MISSING:
            kernel = self.kernel
        if isinstance(kernel, BaseKeyEnableQuantKernelConfig):
            kernel = kernel.specialize_for(self.key)
        elif isinstance(kernel, KeyEnableConfig):
            kernel = kernel if kernel.is_enabled_for(self.key) else None
        assert isinstance(kernel, (BaseQuantKernel, BaseQuantKernelConfig, type(None)))
        
        return super().quantize(
            tensor,
            kernel=kernel,
            channels_dim=channels_dim,
            scale=scale,
            zero=zero,
            dynamic_range=dynamic_range,
            range_bound=range_bound,
            quant_range=quant_range,
            return_with_dequant=return_with_dequant,
            return_with_quant=return_with_quant,
            default_dtype=default_dtype,
            develop_dtype=develop_dtype,
            **kwargs,
        )

    def update(
        self,
        tensor_shape: torch.Size,
        default_dtype: torch.dtype | _MISSING_TYPE = MISSING,
        quant_range: QuantRange | None | _MISSING_TYPE = MISSING,
        range_bound: RangeBound | None | _MISSING_TYPE = MISSING,
    ) -> QuantInfo | None:
        """更新量化信息。
        
        根据张量形状和配置更新内部的量化信息缓存。

        Args:
            tensor_shape (`torch.Size`):
                张量的形状。
            default_dtype: 缩放因子的默认数据类型。
            quant_range: 量化范围约束。
            range_bound: 范围边界约束。

        Returns:
            `QuantInfo` or `None`:
                更新后的量化信息。如果量化器被禁用，返回None。
        """
        return super().update(
            tensor_shape,
            default_dtype=self.default_dtype if default_dtype is MISSING else default_dtype,
            quant_range=self.quant_range if quant_range is MISSING else quant_range,
            range_bound=self.range_bound if range_bound is MISSING else range_bound,
        )

    def quantize_with_low_rank(
        self,
        tensors: torch.Tensor | tp.Sequence[torch.Tensor],
        /,
        *,
        return_with_dequant: bool = True,
        return_with_quant: bool = False,
        kernel: (
            BaseKeyEnableQuantKernelConfig | BaseQuantKernelConfig | BaseQuantKernel | None | _MISSING_TYPE
        ) = MISSING,
        channels_dim: int | None | _MISSING_TYPE = MISSING,
        # scale-based quantization arguments
        scale: torch.Tensor | tp.Sequence[torch.Tensor] | None | _MISSING_TYPE = MISSING,
        zero: torch.Tensor | None | _MISSING_TYPE = MISSING,
        # range-based quantization arguments
        dynamic_range: DynamicRange | tp.Sequence[DynamicRange] | None | _MISSING_TYPE = MISSING,
        range_bound: RangeBound | None | _MISSING_TYPE = MISSING,
        # other arguments
        quant_range: QuantRange | None | _MISSING_TYPE = MISSING,
        default_dtype: torch.dtype | None | _MISSING_TYPE = MISSING,
        develop_dtype: torch.dtype | _MISSING_TYPE = MISSING,
        **kwargs,
    ) -> tuple[list[QuantTensor], list[LowRankBranch] | None]:
        """使用低秩分支补偿进行量化。
        
        这是SVDQuant算法的核心实现。通过低秩分支吸收量化误差，
        可以在保持低位宽的同时获得更好的精度。
        
        工作流程:
            1. 如果启用补偿模式：先量化，然后用低秩分支拟合残差
            2. 如果禁用补偿模式：先提取低秩成分，然后量化剩余部分

        Args:
            tensors: 要量化的张量或张量列表。
            return_with_dequant: 是否返回反量化后的张量。
            return_with_quant: 是否返回量化后的整数张量。
            kernel: 量化核心配置。
            channels_dim: 通道维度。
            scale: 预设的缩放因子。
            zero: 预设的零点。
            dynamic_range: 动态范围。
            range_bound: 范围边界。
            quant_range: 量化范围。
            default_dtype: 缩放因子的默认数据类型。
            develop_dtype: 计算时使用的数据类型。
            **kwargs: 传递给量化核心的其他参数。

        Returns:
            tuple: (量化张量列表, 低秩分支列表或None)
        """
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        
        # 构建量化参数字典
        qkwargs = dict(
            return_with_dequant=return_with_dequant,
            return_with_quant=return_with_quant,
            kernel=kernel,
            channels_dim=channels_dim,
            scale=scale,
            zero=zero,
            dynamic_range=dynamic_range,
            range_bound=range_bound,
            quant_range=quant_range,
            default_dtype=default_dtype,
            develop_dtype=develop_dtype,
            **kwargs,
        )
        
        if self.is_enabled_low_rank():
            qtensors: list[QuantTensor] = []
            branches: list[LowRankBranch] = []
            
            # 独占模式或单个张量：每个张量独立处理
            if len(tensors) == 1 or self.low_rank.exclusive:
                if self.low_rank.compensate:
                    # 补偿模式：先量化，再用低秩分支拟合残差
                    qkwargs["return_with_dequant"] = True
                    for t in tensors:
                        qt = self.quantize(t.data, **qkwargs)
                        lb = LowRankBranch(t.shape[1], t.shape[0], rank=self.low_rank.rank, weight=t.data - qt.data)
                        qtensors.append(qt)
                        branches.append(lb)
                else:
                    # 非补偿模式：先提取低秩成分，再量化剩余部分
                    for t in tensors:
                        lb = LowRankBranch(t.shape[1], t.shape[0], rank=self.low_rank.rank, weight=t.data)
                        qt = self.quantize(t.data - lb.get_effective_weight().view(t.data.shape), **qkwargs)
                        qtensors.append(qt)
                        branches.append(lb)
                return qtensors, branches
            else:
                # 共享模式：多个张量共享低秩分支的A矩阵
                st = torch.cat([t.data for t in tensors], dim=0)
                if self.low_rank.compensate:
                    qkwargs["return_with_dequant"] = True
                    for t in tensors:
                        qt = self.quantize(t.data, **qkwargs)
                        qtensors.append(qt)
                    # 对拼接后的残差进行低秩分解
                    sl = LowRankBranch(
                        st.shape[1],
                        st.shape[0],
                        rank=self.low_rank.rank,
                        weight=st - torch.cat([q.data for q in qtensors], dim=0),
                    )
                    del st
                    # 分割低秩分支
                    i = 0
                    for t in tensors:
                        lb = LowRankBranch(t.shape[1], t.shape[0], rank=self.low_rank.rank)
                        lb.a = sl.a  # 共享A矩阵
                        lb.b.to(dtype=t.dtype, device=t.device)
                        lb.b.weight.copy_(sl.b.weight[i : i + t.shape[0]])
                        branches.append(lb)
                        i += t.shape[0]
                    return qtensors, branches
                else:
                    # 非补偿模式：先对拼接张量提取低秩成分
                    sl = LowRankBranch(st.shape[1], st.shape[0], rank=self.low_rank.rank, weight=st)
                    del st
                    i = 0
                    for t in tensors:
                        lb = LowRankBranch(t.shape[1], t.shape[0], rank=self.low_rank.rank)
                        lb.a = sl.a
                        lb.b.to(dtype=t.dtype, device=t.device)
                        lb.b.weight.copy_(sl.b.weight[i : i + t.shape[0]])
                        qt = self.quantize(t.data - lb.get_effective_weight(), **qkwargs)
                        qtensors.append(qt)
                        branches.append(lb)
                        i += t.shape[0]
                    return qtensors, branches
        else:
            # 未启用低秩分支：普通量化
            return [self.quantize(t.data, **qkwargs) for t in tensors], None

    def state_dict(self, device: torch.device | str = "cpu") -> dict[str, tp.Any]:
        """获取量化器的状态字典。
        
        用于保存量化器的状态，包括缩放因子、零点等。

        Args:
            device: 存储状态字典的设备。

        Returns:
            `dict[str, Any]`:
                状态字典。
        """
        state_dict = {}

        def _copy_to(x):
            return x.to(device).clone()

        state_dict["channels_dim"] = self.channels_dim
        state_dict["scale"] = tree_map(_copy_to, self.scale)
        state_dict["zero"] = _copy_to(self.zero) if self.zero is not None else None
        if self.dynamic_range is None:
            state_dict["dynamic_range"] = None
        elif isinstance(self.dynamic_range, DynamicRange):
            state_dict["dynamic_range"] = tree_map(_copy_to, self.dynamic_range.to_dict())
        else:
            state_dict["dynamic_range"] = tree_map(_copy_to, tuple(d.to_dict() for d in self.dynamic_range))
        state_dict["range_bound"] = self.range_bound.to_dict() if self.range_bound is not None else None
        state_dict["quant_range"] = self.quant_range.to_dict() if self.quant_range is not None else None
        return state_dict

    def load_state_dict(self, state_dict: dict[str, tp.Any], device: torch.device | str = "cpu"):
        """加载状态字典。
        
        从保存的状态字典恢复量化器的状态。

        Args:
            state_dict: 状态字典。
            device: 加载状态字典的目标设备。
        """

        def _move_to(x):
            return x.to(device)

        self.channels_dim = state_dict["channels_dim"]
        self.scale = tree_map(_move_to, state_dict["scale"])
        self.zero = _move_to(state_dict["zero"]) if state_dict["zero"] is not None else None
        if state_dict["dynamic_range"] is None:
            self.dynamic_range = None
        elif isinstance(state_dict["dynamic_range"], dict):
            self.dynamic_range = DynamicRange.from_dict(tree_map(_move_to, state_dict["dynamic_range"]))
        else:
            self.dynamic_range = tuple(
                DynamicRange.from_dict(tree_map(_move_to, d)) for d in state_dict["dynamic_range"]
            )
        self.range_bound = RangeBound.from_dict(state_dict["range_bound"])
        self.quant_range = QuantRange.from_dict(state_dict["quant_range"])
