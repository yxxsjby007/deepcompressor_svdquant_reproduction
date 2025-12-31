# -*- coding: utf-8 -*-
"""量化器配置模块。

本模块定义了量化器的配置类，用于指定量化的各种参数和策略。
配置类采用层次化设计，支持从简单配置到复杂的多级量化配置。

配置层次:
    1. BaseQuantizerConfig: 基础配置抽象类
    2. QuantizerConfig: 标准量化配置
    3. ProgressiveQuantizerConfig: 渐进式多级量化配置
    4. DecomposedQuantizerConfig: 分解后的配置（用于执行）

主要参数:
    - dtype: 量化数据类型（如INT4、FP8等）
    - zero_point: 零点域（对称/非对称量化）
    - group_shapes: 分组形状（用于分组量化）
    - scale_dtypes: 缩放因子的数据类型

使用示例:
    >>> # 创建4位整数量化配置
    >>> config = QuantizerConfig(
    ...     dtype=QuantDataType.from_str("sint4"),
    ...     group_shapes=((-1, -1, 128),),  # 每128个元素一组
    ... )
"""

import typing as tp
from abc import abstractmethod
from dataclasses import dataclass, field

import omniconfig
import torch
from omniconfig import configclass

from ...data.dtype import QuantDataType
from ...data.utils import DtypeUtils, ScaleUtils, ShapeUtils
from ...data.zero import ZeroPointDomain
from ...utils.config import EnableConfig

__all__ = [
    "BaseQuantizerConfig",
    "DecomposedQuantizerConfig",
    "QuantizerConfig",
    "ProgressiveQuantizerConfig",
]


class BaseQuantizerConfig(EnableConfig):
    """量化器基础配置类。
    
    定义了所有量化配置必须实现的接口。
    继承自EnableConfig，支持启用/禁用控制。
    """

    @property
    @abstractmethod
    def quant_dtype(self) -> QuantDataType | None:
        """量化数据类型。None表示不进行量化。"""
        ...

    @property
    @abstractmethod
    def zero_domain(self) -> ZeroPointDomain | None:
        """零点域。None表示对称量化，否则为非对称量化。"""
        ...

    @property
    @abstractmethod
    def largest_group_shape(self) -> tp.Sequence[int]:
        """最大分组形状（最粗粒度的分组）。"""
        ...

    @property
    @abstractmethod
    def smallest_group_shape(self) -> tp.Sequence[int]:
        """最小分组形状（最细粒度的分组）。"""
        ...

    def is_enabled(self) -> bool:
        """检查量化配置是否启用。
        
        Returns:
            如果quant_dtype不为None，返回True。
        """
        return self.quant_dtype is not None

    @abstractmethod
    def decompose(self) -> "DecomposedQuantizerConfig":
        """将配置分解为简单配置列表。
        
        用于将复杂的多级量化配置分解为可执行的步骤序列。
        
        Returns:
            分解后的配置对象。
        """
        ...

    def generate_dirnames(
        self,
        *,
        prefix: str = "",
        shape: torch.Size | tuple[int, ...] = (4096, 4096),
        default_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> list[str]:
        """生成量化配置的目录名称。
        
        用于保存量化模型时生成有意义的目录结构。

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                目录名称前缀。
            shape (`torch.Size` or `tuple[int, ...]`, *optional*, defaults to `(4096, 4096)`):
                待量化张量的形状。
            default_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
                默认数据类型。

        Returns:
            `list[str]`:
                量化配置的名称列表，包含：
                    - 有效位数
                    - 量化数据类型名称
                    - 分组形状名称
        """
        return self.decompose().generate_dirnames(
            prefix=prefix, shape=torch.Size(shape), default_dtype=default_dtype, **kwargs
        )


@dataclass(frozen=True)
class DecomposedQuantizerConfig(BaseQuantizerConfig):
    """分解后的量化器配置类。
    
    将复杂的量化配置分解为一系列简单的量化步骤。
    这是量化执行时实际使用的配置格式。
    
    Attributes:
        steps: 量化步骤配置元组，按执行顺序排列。
        needs_dequant_saturation: 反量化时是否需要饱和处理。
            用于多级量化中防止数值溢出。
    
    Example:
        >>> # 两级量化：先量化到INT8，��量化到INT4
        >>> step1 = QuantizerConfig(dtype=QDType.sint8, ...)
        >>> step2 = QuantizerConfig(dtype=QDType.sint4, ...)
        >>> decomposed = DecomposedQuantizerConfig(steps=(step1, step2))
    """
    
    steps: tuple["QuantizerConfig", ...]  # 量化步骤序列
    needs_dequant_saturation: bool = False  # 是否需要反量化饱和

    @property
    def quant_dtype(self) -> QuantDataType | None:
        """最终的量化数据类型（最后一步的数据类型）。"""
        return self.steps[-1].dtype if self.steps else None

    @property
    def zero_domain(self) -> ZeroPointDomain | None:
        """最终的零点域（最后一步的零点域）。"""
        return self.steps[-1].zero_point if self.steps else None

    @property
    def largest_group_shape(self) -> tp.Sequence[int]:
        """最大分组形状（第一步的分组形状）。"""
        return self.steps[0].largest_group_shape if self.steps else (-1, -1, -1)

    @property
    def smallest_group_shape(self) -> tp.Sequence[int]:
        """最小分组形状（最后一步的分组形状）。"""
        return self.steps[-1].smallest_group_shape if self.steps else (-1, -1, -1)

    @property
    def num_steps(self) -> int:
        """量化步骤数量。"""
        return len(self.steps)

    def decompose(self) -> "DecomposedQuantizerConfig":
        """返回自身（已经是分解后的配置）。"""
        return self

    def __eq__(self, value: object) -> bool:
        """判断两个分解配置是否相等。
        
        只比较关键参数：dtype、group_shapes、scale_dtypes。
        """
        if not isinstance(value, DecomposedQuantizerConfig):
            return False
        if self.num_steps != value.num_steps:
            return False
        for rhs, lhs in zip(self.steps, value.steps, strict=True):
            # 只比较关键参数
            if rhs.dtype != lhs.dtype:
                return False
            if rhs.group_shapes != lhs.group_shapes:
                return False
            if rhs.scale_dtypes != lhs.scale_dtypes:
                return False
        if self.num_steps > 1:
            if self.needs_dequant_saturation != value.needs_dequant_saturation:
                return False
        return True

    def _get_effective_bits(
        self, *, shape: torch.Size | tuple[int, ...] = (4096, 4096), default_dtype: torch.dtype = torch.float16
    ) -> float:
        """计算量化的有效位数。
        
        有效位数 = 数据位数 + 缩放因子开销 + 零点开销
        
        Args:
            shape: 待量化张量的形状。
            default_dtype: 默认数据类型。

        Returns:
            有效位数（浮点数，因为包含缩放因子的分摊开销）。
        """
        shape = torch.Size(shape)
        if self.quant_dtype is None:
            return DtypeUtils.infer_dtype_bits(default_dtype)
        bits = self.quant_dtype.total_bits
        # 累加每个步骤的缩放因子开销
        for step_config in self.steps:
            group_shapes = ShapeUtils.infer_group_shapes(step_config.group_shapes, shape=shape)
            scale_dtypes = ScaleUtils.infer_scale_dtypes(step_config.scale_dtypes, default_dtype=default_dtype)
            for group_shape, scale_dtype in zip(group_shapes, scale_dtypes, strict=True):
                # 缩放因子开销 = 缩放因子位数 / 组大小
                bits += DtypeUtils.infer_dtype_bits(scale_dtype) / group_shape.numel()
        # 添加零点开销
        if self.zero_domain == ZeroPointDomain.PreScale:
            bits += self.quant_dtype.total_bits / group_shapes[-1].numel()
        elif self.zero_domain == ZeroPointDomain.PostScale:
            bits += DtypeUtils.infer_dtype_bits(scale_dtype) / group_shape.numel()
        return bits

    def _get_dtype_name(self, default_dtype: torch.dtype = torch.float16) -> str:
        """获取量化数据类型的名称字符串。

        Args:
            default_dtype: 默认数据类型。

        Returns:
            数据类型名称，包含零点域后缀（如 "sint4.z"）。
        """
        if self.quant_dtype is None:
            return DtypeUtils.infer_dtype_name(default_dtype)
        name = DtypeUtils.infer_dtype_name(self.quant_dtype)
        if self.zero_domain == ZeroPointDomain.PreScale:
            name += ".z"  # 预缩放零点
        elif self.zero_domain == ZeroPointDomain.PostScale:
            name += ".zp"  # 后缩放零点
        return name

    def _get_group_shapes_name(self, default_dtype: torch.dtype = torch.float16) -> str:
        """获取分组形状的名称字符串。

        Args:
            default_dtype: 默认数据类型。

        Returns:
            分组形状名称，格式如 "g128.fp16.g32.sint8"。
        """
        if self.quant_dtype is None:
            return f"tnsr.{DtypeUtils.infer_dtype_name(default_dtype)}"
        num_steps = len(self.steps)
        names = []
        step_default_dtype = default_dtype
        for step, step_config in enumerate(self.steps):
            step_names = []
            for group_shape, sdtype in zip(step_config.group_shapes, step_config.scale_dtypes, strict=True):
                name = f"{ShapeUtils.infer_group_shape_name(group_shape)}"
                name += f".{DtypeUtils.infer_dtype_name(sdtype or step_default_dtype)}"
                step_names.append(name)
            step_name = ".".join(reversed(step_names))
            names.append(f"[{step_name}]" if step < num_steps - 2 else step_name)
            step_default_dtype = step_config.dtype
            assert step_default_dtype is not None, "step_default_dtype must not be None"
        return ".".join(reversed(names))

    def generate_dirnames(
        self,
        *,
        prefix: str = "",
        shape: torch.Size | tuple[int, ...] = (4096, 4096),
        default_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> list[str]:
        """生成量化配置的目录名称列表。

        Args:
            prefix: 目录名称前缀。
            shape: 待量化张量的形状。
            default_dtype: 默认数据类型。

        Returns:
            目录名称列表：[有效位数, 数据类型名, 分组形状名]
        """
        shape = torch.Size(shape)
        bits_str = str(int(self._get_effective_bits(shape=shape, default_dtype=default_dtype)))
        dtype_str = self._get_dtype_name(default_dtype=default_dtype)
        group_str = self._get_group_shapes_name(default_dtype=default_dtype)
        names = [bits_str, dtype_str, group_str]
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names


@configclass
@dataclass
class QuantizerConfig(BaseQuantizerConfig):
    """Quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        zero_point (`ZeroPointDomain` or `None`, *optional*, defaults to `None`):
            The zero-point domain.
        group_shapes (`Sequence[Sequence[int]]`, *optional*, defaults to `((-1, -1, -1),)`):
            The shapes for per-group quantization.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`, *optional*, defaults to `(None,)`):
            The quantization scale data type for per-group quantization.
    """

    dtype: QuantDataType | None = None
    zero_point: ZeroPointDomain | None = None
    group_shapes: tp.Sequence[tp.Sequence[int]] = field(
        default=((-1, -1, -1),),
        metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": lambda s: [int(n) for n in s.split(",")]}},
    )
    scale_dtypes: tp.Sequence[torch.dtype | QuantDataType | None] = field(
        default=(None,), metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": DtypeUtils.eval_dtype}}
    )

    def __post_init__(self) -> None:
        self.group_shapes, self.scale_dtypes = ShapeUtils.format_group_configs(
            group_shapes=self.group_shapes, scale_dtypes=self.scale_dtypes
        )
        if self.dtype is None:
            self.group_shapes, self.scale_dtypes = ((-1, -1, -1),), (None,)

    @property
    def quant_dtype(self) -> QuantDataType | None:
        """The final quantization data type."""
        return self.dtype

    @property
    def zero_domain(self) -> ZeroPointDomain | None:
        """The final zero-point domain."""
        return self.zero_point

    @property
    def largest_group_shape(self) -> tp.Sequence[int]:
        """The shape of the largest group."""
        return self.group_shapes[0]

    @property
    def smallest_group_shape(self) -> tp.Sequence[int]:
        """The shape of the smallest group."""
        return self.group_shapes[-1]

    def decompose(self) -> DecomposedQuantizerConfig:
        """Decompose the configuration to a list of simple configurations."""
        return DecomposedQuantizerConfig(steps=(self,) if self.dtype is not None else ())


@configclass
@dataclass
class ProgressiveQuantizerConfig(QuantizerConfig):
    """Progressive Quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        zero_point (`ZeroPointDomain` or `None`, *optional*, defaults to `None`):
            The zero-point domain.
        group_shapes (`Sequence[Sequence[int]]`, *optional*, defaults to `((-1, -1, -1),)`):
            The shapes for per-group quantization.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`, *optional*, defaults to `(None,)`):
            The quantization scale data type for per-group quantization.
        intermediate_dtypes (`Sequence[QuantDataType]`, *optional*, defaults to `()`):
            The intermediate quantization data types.
        intermediate_levels (Sequence[int], *optional*, defaults to `()`):
            The intermediate quantization levels.
        needs_dequant_saturation (`bool`, *optional*, defaults to `False`):
            Whether the dequantization needs saturation.
    """

    intermediate_dtypes: tp.Sequence[QuantDataType] = field(
        default_factory=tuple, metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": QuantDataType.from_str}}
    )
    intermediate_levels: tp.Sequence[int] = field(
        default_factory=tuple, metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": int}}
    )
    needs_dequant_saturation: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dtype is None:
            self.intermediate_dtypes = ()
            self.intermediate_levels = ()
            self.needs_dequant_saturation = False
            return
        num_levels = len(self.group_shapes)
        if isinstance(self.intermediate_dtypes, QuantDataType):
            self.intermediate_dtypes = (self.intermediate_dtypes,)
        if isinstance(self.intermediate_levels, int):
            self.intermediate_levels = (self.intermediate_levels,)
        self.intermediate_dtypes = tuple(self.intermediate_dtypes)
        self.intermediate_levels = tuple(level % num_levels for level in self.intermediate_levels)
        if len(self.intermediate_dtypes) == 0:
            self.intermediate_levels = ()
            self.needs_dequant_saturation = False
        assert len(self.intermediate_dtypes) == len(self.intermediate_levels)
        assert len(self.intermediate_levels) < num_levels
        assert all(isinstance(dtype, QuantDataType) for dtype in self.intermediate_dtypes)
        assert all(level < num_levels - 1 for level in self.intermediate_levels)

    def decompose(self) -> DecomposedQuantizerConfig:
        """Decompose the configuration to a list of simple configurations."""
        if self.dtype is None:
            return DecomposedQuantizerConfig(steps=())
        elif len(self.intermediate_dtypes) == 0:
            return DecomposedQuantizerConfig(steps=(self,))
        else:
            steps = []
            prev_level = 0
            for level, dtype in zip(self.intermediate_levels, self.intermediate_dtypes, strict=True):
                steps.append(
                    QuantizerConfig(
                        dtype=dtype,
                        zero_point=None,
                        group_shapes=self.group_shapes[prev_level : level + 1],
                        scale_dtypes=self.scale_dtypes[prev_level : level + 1],
                    )
                )
                prev_level = level + 1
            steps.append(
                QuantizerConfig(
                    dtype=self.dtype,
                    zero_point=self.zero_point,
                    group_shapes=self.group_shapes[prev_level:],
                    scale_dtypes=self.scale_dtypes[prev_level:],
                )
            )
            return DecomposedQuantizerConfig(steps=tuple(steps), needs_dequant_saturation=self.needs_dequant_saturation)
