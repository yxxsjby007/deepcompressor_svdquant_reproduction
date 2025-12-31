# -*- coding: utf-8 -*-
"""量化信息模块。

本模块定义了量化过程中使用的信息类，用于管理和缓存量化参数。
这些信息类在量化执行前计算，避免重复计算开销。

主要组件:
    - QuantStepInfo: 单步量化信息，包含一个量化步骤的所有参数
    - QuantInfo: 完整量化信息，包含多步量化的所有参数

信息类的作用:
    1. 预计算量化参数（视图形状、分组形状等）
    2. 缓存计算结果，避免重复计算
    3. 检测配置变化，按需更新
"""

from dataclasses import dataclass, field

import torch

from ...data.dtype import QuantDataType
from ...data.range import ProtectiveQuantRange, QuantRange, RangeBound
from ...data.utils import ShapeUtils
from ...data.zero import ZeroPointDomain
from ..config.base import DecomposedQuantizerConfig, QuantizerConfig
from .scale import QuantScaleInfo

__all__ = ["QuantScaleInfo", "QuantStepInfo", "QuantInfo"]


@dataclass
class QuantStepInfo:
    """单步量化信息类。
    
    包含执行一个量化步骤所需的所有信息，包括：
    - 量化配置（数据类型、零点域、分组形状等）
    - 张量形状信息（原始形状、视图形状、分组形状）
    - 缩放因子信息
    
    Attributes:
        quant_dtype: 量化数据类型
        zero_domain: 零点域
        group_shapes: 分组形状配置
        scale_dtypes: 缩放因子数据类型
        quant_range: 量化范围约束
        range_bound: 范围边界约束
        default_dtype: 默认数据类型
        tensor_shape: 张量原始形状
        tensor_group_shapes: 张量的分组形状列表
        tensor_view_shape: 张量的视图形状
        scale: 缩放因子信息对象
    """
    
    # region 配置参数
    quant_dtype: QuantDataType              # 量化数据类型
    zero_domain: ZeroPointDomain | None     # 零点域
    group_shapes: tuple[tuple[int, ...], ...]  # 分组形状配置
    scale_dtypes: tuple[torch.dtype | QuantDataType | None, ...]  # 缩放因子数据类型
    quant_range: QuantRange | None          # 量化范围约束
    range_bound: RangeBound | None          # 范围边界约束
    default_dtype: torch.dtype              # 默认数据类型
    # endregion
    
    # region 形状信息
    tensor_shape: torch.Size
    """张量的原始形状，格式为 (s0, s1, ...)"""
    
    tensor_group_shapes: list[torch.Size]
    """每层分组的形状列表，每个形状格式为 (gs0, gs1, ...)"""
    
    tensor_view_shape: torch.Size
    """张量的视图形状，格式为 (#g0, gs0, #g1, gs1, ...)
    其中 #gi 是第i维的分组数，gsi 是组大小"""
    # endregion
    
    scale: QuantScaleInfo = field(init=False)
    """缩放因子信息对象"""

    def __post_init__(self):
        """初始化后处理：创建缩放因子信息对象。"""
        self.scale = QuantScaleInfo(
            tensor_view_shape=self.tensor_view_shape,
            tensor_quant_dtype=self.quant_dtype,
            tensor_zero_domain=self.zero_domain,
            tensor_quant_range=self.quant_range,
            tensor_range_bound=self.range_bound,
            scale_view_shapes=ShapeUtils.infer_scale_view_shapes(self.tensor_group_shapes, shape=self.tensor_shape),
            scale_quant_dtypes=self.scale_dtypes,
            default_quant_dtype=self.default_dtype,
        )

    @property
    def tensor_zero_domain(self) -> ZeroPointDomain | None:
        """张量的零点域。"""
        return self.scale.tensor_zero_domain

    @property
    def tensor_quant_range(self) -> QuantRange:
        """张量的量化范围（与数据类型范围的交集）。"""
        return self.scale.tensor_quant_range

    @property
    def tensor_range_bound(self) -> RangeBound | None:
        """张量的范围边界约束。"""
        return self.scale.tensor_range_bound

    def to_config(self) -> QuantizerConfig:
        """将信息转换回配置对象。
        
        Returns:
            对应的QuantizerConfig配置对象。
        """
        return QuantizerConfig(
            dtype=self.quant_dtype,
            zero_point=self.zero_domain,
            group_shapes=self.tensor_group_shapes,
            scale_dtypes=self.scale.scale_quant_dtypes,
        )

    @staticmethod
    def construct(
        config: QuantizerConfig,
        tensor_shape: torch.Size,
        default_dtype: torch.dtype,
        quant_range: QuantRange | None = None,
        range_bound: RangeBound | None = None,
    ) -> "QuantStepInfo":
        """从配置构建量化步骤信息。
        
        Args:
            config: 量化配置对象。
            tensor_shape: 张量的形状。
            default_dtype: 默认数据类型。
            quant_range: 量化范围约束。
            range_bound: 范围边界约束。
            
        Returns:
            构建的QuantStepInfo对象。
        """
        # 推断分组形状
        tensor_group_shapes = ShapeUtils.infer_group_shapes(config.group_shapes, shape=tensor_shape)
        # 推断视图形状（用于分组量化）
        tensor_view_shape = ShapeUtils.infer_view_shape(tensor_shape, group_shape=tensor_group_shapes[-1])
        
        return QuantStepInfo(
            quant_dtype=config.dtype,
            zero_domain=config.zero_point,
            group_shapes=config.group_shapes,
            scale_dtypes=config.scale_dtypes,
            quant_range=quant_range,
            range_bound=range_bound,
            default_dtype=default_dtype,
            tensor_shape=tensor_shape,
            tensor_group_shapes=tensor_group_shapes,
            tensor_view_shape=tensor_view_shape,
        )


@dataclass
class QuantInfo:
    """完整量化信息类。
    
    包含多步量化的所有信息。对于渐进式量化，可能包含多个量化步骤。
    
    Attributes:
        steps: 量化步骤信息元组
        needs_dequant_saturation: 反量化时是否需要饱和处理
        
    Example:
        >>> # 单步量化
        >>> info = QuantInfo.construct(config, tensor_shape, default_dtype)
        >>> print(info.num_steps)  # 1
        >>> 
        >>> # 多步渐进量化
        >>> info = QuantInfo.construct(progressive_config, tensor_shape, default_dtype)
        >>> print(info.num_steps)  # 2 或更多
    """
    
    steps: tuple[QuantStepInfo, ...]  # 量化步骤信息序列
    needs_dequant_saturation: bool = False  # 是否需要反量化饱和

    @property
    def num_steps(self) -> int:
        """量化步骤数量。"""
        return len(self.steps)

    def get_child(self, idx: int) -> QuantStepInfo:
        """获取指定索引的量化步骤信息。
        
        Args:
            idx: 步骤索引。
            
        Returns:
            对应的QuantStepInfo对象。
        """
        return self.steps[idx]

    def is_outdated(
        self,
        config: DecomposedQuantizerConfig,
        tensor_shape: torch.Size,
        default_dtype: torch.dtype,
        quant_range: QuantRange | None = None,
        range_bound: RangeBound | None = None,
    ) -> bool:
        """检查当前量化信息是否过时。
        
        当配置或张量形状发生变化时，需要重新计算量化信息。
        
        Args:
            config: 当前的量化配置。
            tensor_shape: 当前的张量形状。
            default_dtype: 当前的默认数据类型。
            quant_range: 当前的量化范围约束。
            range_bound: 当前的范围边界约束。
            
        Returns:
            如果信息过时需要更新，返回True。
        """
        # 检查步骤数量
        if self.num_steps != config.num_steps:
            return True
        
        # 检查每个步骤的关键参数
        for step_info, step_config in zip(self.steps, config.steps, strict=True):
            if step_info.quant_dtype != step_config.quant_dtype:
                return True
            if step_info.group_shapes != step_config.group_shapes:
                return True
            if step_info.scale_dtypes != step_config.scale_dtypes:
                return True
        
        # 检查全局参数
        if self.num_steps > 0:
            first_step = self.steps[0]
            if first_step.tensor_shape != tensor_shape:
                return True
            if first_step.default_dtype != default_dtype:
                return True
            if first_step.range_bound != range_bound:
                return True
            if self.steps[-1].quant_range != quant_range:
                return True
            if self.num_steps > 1 and self.needs_dequant_saturation != config.needs_dequant_saturation:
                return True
        
        return False

    @staticmethod
    def construct(
        config: DecomposedQuantizerConfig,
        tensor_shape: torch.Size,
        default_dtype: torch.dtype,
        quant_range: QuantRange | None = None,
        range_bound: RangeBound | None = None,
    ) -> "QuantInfo":
        """从分解后的配置构建量化信息。
        
        对于多步量化，会自动处理中间步骤的保护性量化范围，
        确保中间结果不会超出后续步骤的表示范围。
        
        Args:
            config: 分解后的量化配置。
            tensor_shape: 张量的形状。
            default_dtype: 默认数据类型。
            quant_range: 最终的量化范围约束。
            range_bound: 范围边界约束。
            
        Returns:
            构建的QuantInfo对象。
        """
        steps: list[QuantStepInfo] = []
        num_steps = config.num_steps
        step_default_dtype = default_dtype
        step_range_bound = range_bound
        
        for step, step_config in enumerate(config.steps):
            assert step_config.quant_dtype is not None, f"quant_dtype is required for step {step}"
            
            # 确定当前步骤的量化范围
            if step == num_steps - 1:
                # 最后一步：使用用户指定的量化范围
                step_quant_range = quant_range
            elif step < num_steps - 2 or config.needs_dequant_saturation:
                # 非倒数第二步，或需要饱和处理：不使用保护性范围
                step_quant_range = None
            else:
                # 倒数第二步且不需要饱和：使用保护性量化范围
                # 确保量化结果不会超出最后一步的表示范围
                step_quant_range = ProtectiveQuantRange.construct(
                    outer_dtype=step_config.quant_dtype,
                    inner_dtype=config.steps[-1].quant_dtype,
                    zero_domain=config.steps[-1].zero_domain,
                    inner_quant_range=quant_range,
                )
            
            # 构建当前步骤的信息
            steps.append(
                QuantStepInfo.construct(
                    step_config,
                    tensor_shape=tensor_shape,
                    default_dtype=step_default_dtype,
                    quant_range=step_quant_range,
                    range_bound=step_range_bound,
                )
            )
            
            # 更新下一步的默认类型和范围边界
            step_default_dtype = step_config.quant_dtype
            step_range_bound = steps[-1].scale.tensor_quant_range
        
        return QuantInfo(steps=tuple(steps), needs_dequant_saturation=config.needs_dequant_saturation)
