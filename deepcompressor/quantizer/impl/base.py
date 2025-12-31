# -*- coding: utf-8 -*-
"""量化器实现模块。

本模块定义了量化器的核心实现类QuantizerImpl，它是所有量化操作的基础。
QuantizerImpl封装了完整的量化流程，包括：
- 缩放因子和零点的计算
- 多步渐进量化
- 量化和反量化操作

量化流程概述:
    1. 更新量化信息（如果配置或形状发生变化）
    2. 计算每个量化步骤的缩放因子和零点
    3. 使用量化核心（如RTN、GPTQ）执行量化
    4. 执行反量化得到近似值

主要组件:
    - QuantizerImpl: 量化器实现类
"""

import typing as tp
from dataclasses import dataclass, field

import torch

from ...data.range import DynamicRange, QuantRange, RangeBound
from ...data.scale import QuantScale
from ...data.tensor import QuantTensor
from ...data.zero import ZeroPointDomain
from ...utils.config import KeyEnableConfig
from ..config.base import BaseQuantizerConfig
from ..config.kernel import BaseQuantKernel, BaseQuantKernelConfig
from ..kernel.rtn import QuantRtnKernel
from .info import QuantInfo

__all__ = ["QuantizerImpl"]


@dataclass
class QuantizerImpl:
    """量化器实现类。
    
    这是量化操作的核心实现，提供了完整的量化和反量化功能。
    支持多步渐进量化、分组量化、对称/非对称量化等多种模式。

    Args:
        config (`BasicQuantizerConfig` or `None`):
            量化器配置。None表示禁用量化。
        key (`str`, *optional*, defaults to `""`):
            量化器的键名，用于区分不同的量化器实例。

    Attributes:
        info (`QuantInfo` or `None`):
            缓存的量化信息。在配置或张量形状变化时自动更新。
            
    Example:
        >>> impl = QuantizerImpl(config=config)
        >>> if impl.is_enabled():
        ...     qtensor = impl.quantize(weight)
        ...     quantized_weight = qtensor.data
    """

    config: BaseQuantizerConfig | None  # 量化配置
    key: str = ""                        # 量化器键名
    info: QuantInfo | None = field(init=False, default=None)  # 缓存的量化信息

    def is_enabled(self) -> bool:
        """检查量化器是否启用。
        
        Returns:
            如果量化器配置存在且启用，返回True。
        """
        if self.config is None:
            return False
        if isinstance(self.config, KeyEnableConfig):
            return self.config.is_enabled_for(self.key)
        return self.config.is_enabled()

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        kernel: BaseQuantKernel | BaseQuantKernelConfig | None = None,
        channels_dim: int | None = None,
        # 基于缩放因子的量化参数
        scale: torch.Tensor | tp.Sequence[torch.Tensor] | None = None,
        zero: torch.Tensor | None = None,
        # 基于范围的量化参数
        dynamic_range: DynamicRange | tp.Sequence[DynamicRange] | None = None,
        # 其他参数
        range_bound: RangeBound | None = None,
        quant_range: QuantRange | None = None,
        return_with_dequant: bool = True,
        return_with_quant: bool = False,
        default_dtype: torch.dtype | None = torch.float16,
        develop_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> QuantTensor:
        """量化浮点张量。
        
        这是量化的主入口函数，处理通道维度的重塑，然后调用内部量化方法。

        Args:
            tensor (`torch.Tensor`):
                要量化的浮点张量。
            kernel (`QuantKernel` or `QuantKernelConfig` or `None`, *optional*, defaults to `None`):
                量化核心或其配置。None表示使用默认的RTN核心。
            channels_dim (`int` or `None`, *optional*, defaults to `None`):
                通道维度。如果指定，张量会被重塑为 (-1, *shape[channels_dim:])。
            scale (`torch.Tensor` or `Sequence[torch.Tensor]` or `None`, *optional*, defaults to `None`):
                预设的缩放因子张量。如果提供，跳过缩放因子计算。
            zero (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                预设的零点张量。
            dynamic_range (`DynamicRange` or `Sequence[DynamicRange]` or `None`, *optional*, defaults to `None`):
                动态范围配置。用于基于范围的缩放因子计算。
            range_bound (`RangeBound` or `None`, *optional*, defaults to `None`):
                动态范围边界约束。
            quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                量化范围约束。
            return_with_dequant (`bool`, *optional*, defaults to `True`):
                是否返回反量化后的张量。
            return_with_quant (`bool`, *optional*, defaults to `False`):
                是否返回量化后的整数张量。
            default_dtype (`torch.dtype` or `None`, *optional*, defaults to `torch.float16`):
                缩放因子的默认数据类型。
            develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                量化计算时使用的数据类型（通常使用高精度）。
            **kwargs:
                传递给量化核心的其他参数。例如：
                - ``inputs``: GPTQ核心的输入张量
                - ``round_delta``: RTN核心的舍入增量

        Returns:
            `QuantTensor`:
                量化结果对象，包含反量化张量、量化张量、缩放因子等。
        """
        shape = tensor.shape
        # 处理通道维度
        if channels_dim is not None:
            tensor = tensor.reshape(-1, *shape[channels_dim:])
        
        # 处理舍入增量
        round_delta = kwargs.pop("round_delta", None)
        if round_delta is not None:
            round_delta = round_delta.view(-1, *shape[channels_dim:])
        
        # 调用内部量化方法
        result = self._quantize(
            tensor,
            kernel=kernel,
            scale=scale,
            zero=zero,
            dynamic_range=dynamic_range,
            range_bound=range_bound,
            quant_range=quant_range,
            round_delta=round_delta,
            return_with_dequant=return_with_dequant,
            return_with_quant=return_with_quant,
            default_dtype=default_dtype or tensor.dtype,
            develop_dtype=develop_dtype,
            **kwargs,
        )
        
        # 恢复原始形状
        if result.data is not None:
            result._dequantized = result.data.view(shape)
        if result.qdata is not None:
            result._quantized = result.qdata.view(shape)
        return result

    def _quantize(  # noqa: C901
        self,
        tensor: torch.Tensor,
        *,
        kernel: BaseQuantKernel | BaseQuantKernelConfig | None = None,
        # 基于缩放因子的量化参数
        scale: torch.Tensor | tp.Sequence[torch.Tensor | None] | None = None,
        zero: torch.Tensor | None = None,
        # 基于范围的量化参数
        dynamic_range: DynamicRange | tp.Sequence[DynamicRange | None] | None = None,
        # 其他参数
        range_bound: RangeBound | None = None,
        quant_range: QuantRange | None = None,
        round_delta: torch.Tensor | None = None,
        return_with_dequant: bool = True,
        return_with_quant: bool = False,
        default_dtype: torch.dtype = torch.float16,
        develop_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> QuantTensor:
        """内部量化方法。
        
        执行实际的量化操作，包括：
        1. 更新量化信息
        2. 计算缩放因子和零点
        3. 执行量化
        4. 执行反量化

        Args:
            tensor (`torch.Tensor`):
                要量化的浮点张量。
            kernel: 量化核心。
            scale: 预设的缩放因子。
            zero: 预设的零点。
            dynamic_range: 动态范围配置。
            range_bound: 范围边界约束。
            quant_range: 量化范围约束。
            round_delta: 舍入增量。
            return_with_dequant: 是否返回反量化张量。
            return_with_quant: 是否返回量化张量。
            default_dtype: 默认数据类型。
            develop_dtype: 计算数据类型。
            **kwargs: 传递给量化核心的其他参数。

        Returns:
            `QuantTensor`:
                量化结果对象。
        """
        shape, dtype = tensor.shape, tensor.dtype
        
        # 更新量化信息
        self.update(shape, default_dtype, quant_range, range_bound)
        
        # 如果量化被禁用，直接返回原始张量
        if self.info is None or self.info.num_steps == 0:
            return QuantTensor(dequantized=tensor, quantized=tensor, view_shape=shape)
        
        # region 检查scale和dynamic_range参数
        num_steps = self.info.num_steps
        if scale is None:
            scale = (None,) * num_steps
        elif not isinstance(scale, tp.Sequence):
            scale = (scale,)
        if dynamic_range is None:
            dynamic_range = (None,) * num_steps
        elif isinstance(dynamic_range, DynamicRange):
            if not dynamic_range.is_set():
                dynamic_range = (None,) * num_steps
            else:
                dynamic_range = (dynamic_range,)
        assert isinstance(scale, (tuple, list)), "scale must be a tuple or list."
        assert len(scale) == num_steps, "scale must have the same length as infos."
        assert isinstance(dynamic_range, (tuple, list)), "dynamic_range must be a tuple or list."
        assert len(dynamic_range) == num_steps, "dynamic_range must have the same length as infos."
        # endregion
        
        # region 计算并量化每个步骤的缩放因子和零点
        quant_scale = QuantScale()
        develop_tensor = tensor.to(dtype=develop_dtype) if dtype != develop_dtype else tensor.clone()
        
        for step, (step_info, step_scale, step_dynamic_range) in enumerate(
            zip(self.info.steps, scale, dynamic_range, strict=True)
        ):
            # 计算当前步骤的缩放因子和零点
            step_scale, step_zero = step_info.scale.quantize(
                scale=step_scale,
                zero=None if step < num_steps - 1 else zero,  # 只有最后一步使用零点
                tensor=develop_tensor,
                dynamic_range=step_dynamic_range,
            )
            quant_scale.append(step_scale)
            
            # 对于中间步骤，应用缩放并裁剪
            if step < num_steps - 1:
                step_quant_range = step_info.tensor_quant_range
                develop_tensor = develop_tensor.view(step_info.tensor_view_shape).div_(step_scale.data).view(shape)
                develop_tensor = develop_tensor.clamp_(min=step_quant_range.min, max=step_quant_range.max)
        
        quant_zero = step_zero
        # endregion
        
        # region 执行量化
        assert isinstance(step_scale, QuantScale), "The last scale must be a QuantScale."
        assert isinstance(step_zero, torch.Tensor), "The last zero point must be a tensor."
        
        # 处理舍入增量的形状
        if round_delta is not None:
            if round_delta.shape[0] == 1:
                round_delta = round_delta.view(1, 1, *step_info.tensor_view_shape[2:])
            else:
                round_delta = round_delta.view(step_info.tensor_view_shape)
        
        # 构建量化核心
        if isinstance(kernel, BaseQuantKernelConfig):
            kernel = kernel.build()
        kernel = kernel or QuantRtnKernel()  # 默认使用RTN
        
        # 执行量化
        develop_tensor = kernel.quantize(
            tensor=develop_tensor,
            view_shape=step_info.tensor_view_shape,
            quant_dtype=step_info.quant_dtype,
            zero_domain=step_info.zero_domain,
            scale=step_scale.data,
            zero=step_zero,
            quant_range=step_info.quant_range,
            range_bound=step_info.range_bound,
            round_delta=round_delta,
            **kwargs,
        )
        assert not develop_tensor.isnan().any(), "Quantized tensor contains NaN."
        assert not develop_tensor.isinf().any(), "Quantized tensor contains Inf."
        # endregion
        
        # region 处理量化后的张量
        quantized = None
        if return_with_quant:
            quantized = develop_tensor.detach()
            if return_with_dequant:
                quantized = develop_tensor.clone()
            quantized = develop_tensor.view(shape)
        # endregion
        
        # region 执行反量化
        dequantized = None
        if return_with_dequant:
            dequantized = develop_tensor
            
            # 反量化最后一步
            if self.config.zero_domain == ZeroPointDomain.PreScale:
                dequantized = dequantized.sub_(step_zero)
            dequantized = dequantized.mul_(step_scale.data)
            if self.config.zero_domain == ZeroPointDomain.PostScale:
                dequantized = dequantized.sub_(step_zero)
            
            # 反量化中间步骤（从后向前）
            for step in range(num_steps - 2, -1, -1):
                step_info, step_scale = self.info.get_child(step), quant_scale.get_child(step)
                step_min, step_max = step_info.quant_dtype.min_value, step_info.quant_dtype.max_value
                
                # 饱和处理（如果需要）
                if self.info.needs_dequant_saturation or step < num_steps - 2:
                    dequantized = dequantized.clamp_(min=step_min, max=step_max)
                else:
                    assert dequantized.max() <= step_max, "Quantized tensor exceeds maximum value."
                    assert dequantized.min() >= step_min, "Quantized tensor exceeds minimum value."
                
                # 乘以缩放因子
                dequantized = dequantized.view(step_info.tensor_view_shape).mul_(step_scale.data)
            
            dequantized = dequantized.view(shape).to(dtype=dtype)
        # endregion
        
        return QuantTensor(
            dequantized=dequantized,
            quantized=quantized,
            scale=quant_scale if return_with_quant else None,
            zero=quant_zero if return_with_quant else None,
            view_shape=self.info.steps[-1].tensor_view_shape if return_with_quant else None,
        )

    def update(
        self,
        tensor_shape: torch.Size,
        default_dtype: torch.dtype | None,
        quant_range: QuantRange | None,
        range_bound: RangeBound | None,
    ) -> QuantInfo | None:
        """更新量化信息。
        
        当配置或张量形状发生变化时，重新计算量化信息。
        量化信息会被缓存，避免重复计算。

        Args:
            tensor_shape (`torch.Size`):
                张量的形状。
            default_dtype (`torch.dtype` or `None`):
                缩放因子的默认数据类型。
            quant_range (`QuantRange` or `None`):
                量化范围约束。
            range_bound (`RangeBound` or `None`):
                范围边界约束。

        Returns:
            `QuantInfo` or `None`:
                更新后的量化信息。如果量化器被禁用，返回None。
        """
        if not self.is_enabled():
            self.info = None
        else:
            config = self.config.decompose()
            assert default_dtype is not None, "default_dtype must be set."
            # 检查是否需要更新
            if self.info is None or self.info.is_outdated(
                config, tensor_shape, default_dtype, quant_range, range_bound
            ):
                self.info = QuantInfo.construct(
                    config, tensor_shape, default_dtype, quant_range=quant_range, range_bound=range_bound
                )
        return self.info
