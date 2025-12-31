# -*- coding: utf-8 -*-
"""量化动态范围计算模块。

本模块定义了量化过程中用于计算和管理数值范围的数据结构。
正确的范围计算对于量化精度至关重要。

范围类型:
    1. RangeBound: 基础范围边界，定义最小/最大值
    2. QuantRange: 量化范围，用于确定量化参数
    3. LogQuantRange: 对数空间的量化范围，用于浮点量化
    4. ProtectiveQuantRange: 保护性量化范围，防止溢出
    5. DynamicRange: 动态范围，从数据中测量得到

量化范围的作用:
    - 确定缩放因子(scale)的计算
    - 限制量化值的范围
    - 处理异常值(outliers)

主要组件:
    - RangeBound: 范围边界基类
    - QuantRange: 量化范围类
    - LogQuantRange: 对数量化范围类
    - ProtectiveQuantRange: 保护性量化范围类
    - DynamicRange: 动态范围类
"""

import math
import typing as tp
from dataclasses import dataclass

import torch

from .dtype import QuantDataType
from .zero import ZeroPointDomain

__all__ = ["RangeBound", "QuantRange", "LogQuantRange", "ProtectiveQuantRange", "DynamicRange"]


@dataclass
class RangeBound:
    """范围边界数据类。
    
    定义数值范围的最小值和最大值边界。
    
    Attributes:
        min: 最小值边界，None表示无限制
        max: 最大值边界，None表示无限制
    """

    min: float | None = None  # 最小值边界
    max: float | None = None  # 最大值边界

    def is_set(self) -> bool:
        """检查范围边界是否已设置。
        
        Returns:
            如果min或max任一被设置，返回True。
        """
        return self.min is not None or self.max is not None

    def to_dict(self) -> dict[str, tp.Any]:
        """转换为字典表示。
        
        Returns:
            包含min和max的字典。
        """
        return {"min": self.min, "max": self.max}

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any] | None) -> tp.Optional[tp.Self]:
        """从字典创建范围边界。
        
        Args:
            data: 包含min和max的字典，或None。
            
        Returns:
            RangeBound实例或None。
        """
        return cls(min=data["min"], max=data["max"]) if data is not None else None


class QuantRange(RangeBound):
    """量化范围数据类。
    
    继承自RangeBound，添加了量化特定的范围操作。
    用于确定量化参数和限制量化值的范围。
    """

    def log2(self) -> "LogQuantRange":
        """转换为对数空间的量化范围。
        
        Returns:
            对数空间的量化范围。
        """
        log2_abs_min = int(math.log2(min(abs(self.min or 0), abs(self.max or 0))))
        return LogQuantRange(
            min=None,
            max=None if self.max is None else log2_abs_min,
        )

    def intersect(self, quant_dtype: QuantDataType, *, has_zero_point: bool) -> "QuantRange":
        """计算当前范围与数据类型范围的交集。
        
        确保量化范围不超过数据类型的可表示范围。

        Args:
            quant_dtype (`QuantDataType`):
                量化数据类型。
            has_zero_point (`bool`):
                是否使用零点（非对称量化）。

        Returns:
            `QuantRange`:
                交集后的量化范围。
        """
        max_value = quant_dtype.max_value if self.max is None else min(self.max, quant_dtype.max_value)
        min_value = quant_dtype.min_value if self.min is None else max(self.min, quant_dtype.min_value)
        # 对称量化时，确保范围关于0对称
        if quant_dtype.signed and not has_zero_point:
            max_value = min(abs(min_value), abs(max_value))
            min_value = -max_value
        return QuantRange(min=min_value, max=max_value)

    def intersect_log2(self, quant_dtype: QuantDataType) -> "LogQuantRange":
        """计算当前范围与数据类型范围在对数空间的交集。

        Args:
            quant_dtype (`QuantDataType`):
                量化数据类型。

        Returns:
            `LogQuantRange`:
                对数空间的交集范围。
        """
        return self.log2().intersect_log2(quant_dtype)

    @staticmethod
    def construct(
        dtype: QuantDataType, *, has_zero_point: bool, quant_range: tp.Optional["QuantRange"] = None
    ) -> "QuantRange":
        """构建量化范围。
        
        根据数据类型和额外约束构建最终的量化范围。

        Args:
            dtype (`QuantDataType`):
                量化数据类型。
            has_zero_point (`bool`):
                是否使用零点。
            quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                额外的范围约束。

        Returns:
            `QuantRange`:
                构建的量化范围。
        """
        return (quant_range or QuantRange()).intersect(dtype, has_zero_point=has_zero_point)


class LogQuantRange(QuantRange):
    """对数空间的量化范围数据类。
    
    用于浮点量化，在对数空间中定义范围。
    对于浮点格式，指数决定了数值的数量级。
    """

    def log2(self) -> "LogQuantRange":
        """返回自身（已经是对数空间）。
        
        Returns:
            自身。
        """
        return self

    def intersect(self, quant_dtype: QuantDataType, *, has_zero_point: bool) -> "QuantRange":
        """对数范围不支持直接intersect操作。

        Raises:
            NotImplementedError: 总是抛出此异常。
        """
        raise NotImplementedError("LogQuantRange does not support intersect method")

    def intersect_log2(self, quant_dtype: QuantDataType) -> "LogQuantRange":
        """计算当前对数范围与数据类型指数范围的交集。

        Args:
            quant_dtype (`QuantDataType`):
                量化数据类型。

        Returns:
            `LogQuantRange`:
                对数空间的交集范围。
        """
        max_value = (
            quant_dtype.max_exponent_value if self.max is None else min(self.max, quant_dtype.max_exponent_value)
        )
        min_value = (
            quant_dtype.min_exponent_value if self.min is None else max(self.min, quant_dtype.min_exponent_value)
        )
        return LogQuantRange(min=min_value, max=max_value)

    @staticmethod
    def construct(
        dtype: QuantDataType, quant_range: tp.Optional[tp.Union["LogQuantRange", QuantRange]] = None
    ) -> "LogQuantRange":
        """构建对数空间的量化范围。

        Args:
            dtype (`QuantDataType`):
                量化数据类型。
            quant_range (`LogQuantRange` or `QuantRange` or `None`, *optional*, defaults to `None`):
                额外的范围约束。

        Returns:
            `LogQuantRange`:
                对数空间的量化范围。
        """
        return (quant_range or LogQuantRange()).intersect_log2(dtype)


class ProtectiveQuantRange(QuantRange):
    """保护性量化范围类。
    
    用于多级量化场景，确保内层量化后的值不会超出外层量化的范围。
    这对于防止量化溢出非常重要。
    
    应用场景:
        - W4A8KV4量化中，确保4位权重量化后的值在8位激活量化范围内
        - 嵌套量化结构中的范围保护
    """
    
    # 缓存已计算的保护性范围
    _instances: tp.ClassVar[
        dict[tuple[QuantDataType, QuantDataType, tuple[float, float], ZeroPointDomain], "ProtectiveQuantRange"]
    ] = {}

    @staticmethod
    def construct(
        outer_dtype: QuantDataType,
        inner_dtype: QuantDataType,
        zero_domain: ZeroPointDomain | None,
        inner_quant_range: QuantRange | None = None,
    ) -> QuantRange:
        """构建保护性量化范围。
        
        计算一个安全的量化范围，确保内层量化的结果不会超出外层量化的表示范围。

        Args:
            outer_dtype (`QuantDataType`):
                外层量化的数据类型。
            inner_dtype (`QuantDataType`):
                内层量化的数据类型。
            zero_domain (`ZeroPointDomain` or `None`):
                零点域。
            inner_quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                内层量化范围约束。

        Returns:
            `QuantRange`:
                保护性量化范围。
        """
        assert outer_dtype.is_integer, "outer_dtype must be integer data type"
        assert inner_dtype.is_integer, "inner_dtype must be integer data type"
        assert zero_domain is not None or outer_dtype.signed == inner_dtype.signed
        if zero_domain is None:
            return QuantRange.construct(outer_dtype, has_zero_point=False)

        # 计算内层量化范围
        inner_quant_range = QuantRange.construct(inner_dtype, has_zero_point=True, quant_range=inner_quant_range)
        qmax, qmin = int(inner_quant_range.max), int(inner_quant_range.min)  # type: ignore
        key = (outer_dtype, inner_dtype, (qmin, qmax), zero_domain)
        
        if key not in ProtectiveQuantRange._instances:
            # 计算外层量化范围
            outer_quant_range = QuantRange.construct(outer_dtype, has_zero_point=False)
            vrmax, vrmin = int(outer_quant_range.max), int(outer_quant_range.min)  # type: ignore
            qrmax, qrmin = int(inner_dtype.max_value), int(inner_dtype.min_value)
            
            # 搜索所有有效的值范围组合
            vranges: set[tuple[int, int]] = set()
            for vmax in range(0, vrmax + 1):
                for vmin in range(vrmin, vmax + 1):
                    # 计算缩放因子
                    s = round((vmax - vmin) / (qmax - qmin))
                    assert s >= 0, "s must be non-negative"
                    s = 1 if s == 0 else s
                    s = min(s, vrmax)
                    
                    # 根据零点域计算量化/反量化结果
                    if zero_domain == ZeroPointDomain.PreScale:
                        z = max(min(round(qmin - vmin / s), qrmax), qrmin)
                        m = (max(min(round(vmax / s + z), qmax), qmin) - z) * s
                        n = (max(min(round(vmin / s + z), qmax), qmin) - z) * s
                    elif zero_domain == ZeroPointDomain.PostScale:
                        z = max(min(round(qmin * s - vmin), vrmax), vrmin)
                        m = max(min(round((vmax + z) / s), qmax), qmin) * s - z
                        n = max(min(round((vmin + z) / s), qmax), qmin) * s - z
                    else:
                        raise ValueError(f"unsupported zero-point domain {zero_domain}")
                    
                    # 检查结果是否在外层范围内
                    if vrmin <= m <= vrmax and vrmin <= n <= vrmax:
                        vranges.add((vmin, vmax))
            
            # 找到最大的对称保护范围
            found_pmax = None
            for pmax in range(vrmax, 0, -1):
                pmin = -pmax
                valid = True
                for vmax in range(0, pmax + 1):
                    for vmin in range(pmin, vmax + 1):
                        if (vmin, vmax) not in vranges:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    found_pmax = pmax
                    break
            assert found_pmax is not None, "failed to find the protective quantization range"
            ProtectiveQuantRange._instances[key] = ProtectiveQuantRange(min=-found_pmax, max=found_pmax)
        return ProtectiveQuantRange._instances[key]


@dataclass
class DynamicRange:
    """动态范围数据类。
    
    用于从实际数据中测量和计算量化范围。
    支持静态范围（预设的min/max）和动态范围（从数据测量）。
    
    Attributes:
        min: 最小值张量或None
        max: 最大值张量或None
        ratio: 范围缩放比例，用于调整量化范围
    
    Example:
        >>> # 从数据测量动态范围
        >>> data = torch.randn(100, 100)
        >>> drange = DynamicRange.construct(data, zero_domain=None, is_float_point=False)
        >>> print(drange.max)  # 数据的最大绝对值
    """

    min: torch.Tensor | None = None   # 最小值
    max: torch.Tensor | None = None   # 最大值
    ratio: float | torch.Tensor | None = None  # 范围缩放比例

    def __post_init__(self) -> None:
        """初始化后验证。"""
        if self.max is None:
            assert self.min is None, "min must be None if max is None"

    def is_set(self) -> bool:
        """检查动态范围是否已设置。
        
        Returns:
            如果min、max或ratio任一被设置，返回True。
        """
        return self.min is not None or self.max is not None or self.ratio is not None

    def intersect(self, range_bound: RangeBound | None) -> "DynamicRange":
        """计算当前动态范围与给定范围边界的交集。

        Args:
            range_bound (`RangeBound` or `None`):
                范围边界约束。

        Returns:
            `DynamicRange`:
                交集后的动态范围。
        """
        assert self.max is not None, "max must be specified"
        vmax, vmin = self.max, self.min
        if range_bound is not None:
            if range_bound.max is not None:
                vmax = vmax.clamp(max=range_bound.max)
            if vmin is not None and range_bound.min is not None:
                vmin = vmin.clamp(min=range_bound.min)
        return DynamicRange(min=vmin, max=vmax)

    def measure(  # noqa: C901
        self,
        tensors: torch.Tensor | list[torch.Tensor],
        /,
        *,
        zero_domain: ZeroPointDomain | None,
        is_float_point: bool,
    ) -> "DynamicRange":
        """从给定张量测量动态范围。
        
        根据量化类型（对称/非对称、整数/浮点）计算合适的范围。

        Args:
            tensors (`torch.Tensor` or `list[torch.Tensor]`):
                输入张量，形状为 (#g0, gs0, #g1, gs1, ..., #gn, gsn)。
                其中 #gi 是第i维的分组数，gsi 是组大小。
            zero_domain (`ZeroPointDomain` or `None`):
                零点域。None表示对称量化。
            is_float_point (`bool`):
                是否为浮点量化。

        Returns:
            `DynamicRange`:
                测量得到的动态范围。如果max已指定，返回当前对象。
        """
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        
        # 静态范围：直接使用预设值
        if self.ratio is None and self.max is not None:
            tensor = tensors[0]
            shape = torch.Size([s if i % 2 == 0 else 1 for i, s in enumerate(tensor.shape)])
            vmax = self._format_m_(self.max, shape=shape, dtype=tensor.dtype, device=tensor.device)
            vmin = self._format_m_(self.min, shape=shape, dtype=tensor.dtype, device=tensor.device)
        else:
            if self.max is None:
                assert self.min is None, "min must be None if max is None"
            
            # 需要归约的维度（组内维度）
            reduced = list(range(1, tensors[0].ndim, 2))
            
            # region 步骤1: 确定值范围 (vmax 和 vmin)
            if zero_domain is None:
                # 对称量化：只需要最大绝对值
                vmin = None
                vmax = tensors[0].abs().amax(dim=reduced, keepdim=True)
                for tensor in tensors[1:]:
                    vmax = torch.maximum(vmax, tensor.abs().amax(dim=reduced, keepdim=True).to(vmax.device))
            else:
                # 非对称量化：需要最大值和最小值
                vmax = tensors[0].amax(dim=reduced, keepdim=True)
                for tensor in tensors[1:]:
                    vmax = torch.maximum(vmax, tensor.amax(dim=reduced, keepdim=True).to(vmax.device))
                vmin = tensors[0].amin(dim=reduced, keepdim=True)
                for tensor in tensors[1:]:
                    vmin = torch.minimum(vmin, tensor.amin(dim=reduced, keepdim=True).to(vmin.device))
                
                # 浮点量化：使用均值作为零点参考
                if is_float_point:
                    vavg = tensors[0].mean(dim=reduced, keepdim=True)
                    if len(tensors) > 1:
                        for tensor in tensors[1:]:
                            vavg = vavg + tensor.mean(dim=reduced, keepdim=True).to(vavg.device)
                        vavg = vavg / len(tensors)
            # endregion
            
            # region 步骤2: 按 self.ratio 缩放值范围
            if zero_domain is None:
                if self.ratio is not None:
                    vmax = vmax * self.ratio
            else:
                assert vmin is not None, "vmin must be specified"
                if is_float_point:
                    # 浮点量化：以均值为中心对称缩放
                    vmag = torch.maximum(vmax - vavg, vavg - vmin)
                    if self.ratio is not None:
                        vmag = vmag * self.ratio
                    vmax = vavg + vmag
                    vmin = vavg - vmag
                else:
                    # 整数量化：直接缩放
                    if self.ratio is not None:
                        vmin = vmin * self.ratio
                        vmax = vmax * self.ratio
                
                # PreScale零点域：确保范围包含0
                if zero_domain == ZeroPointDomain.PreScale:
                    vmax = vmax.clamp(min=0)
                    vmin = vmin.clamp(max=0)
            # endregion
            
            # region 步骤3: 用 (self.min, self.max) 限制值范围
            if self.max is not None:
                vmax = vmax.clamp(max=self.max.to(vmax.device))
                if vmin is not None and self.min is not None:
                    vmin = vmin.clamp(min=self.min.to(vmin.device))
            # endregion
        
        return DynamicRange(min=vmin, max=vmax)

    def scale(
        self, ratio: float | torch.Tensor, zero_domain: ZeroPointDomain | None, is_float_point: bool
    ) -> "DynamicRange":
        """按比例缩放当前范围，返回新的动态范围。

        Args:
            ratio (`float` or `torch.Tensor`):
                缩放比例。
            zero_domain (`ZeroPointDomain` or `None`):
                零点域。
            is_float_point (`bool`):
                是否为浮点量化。

        Returns:
            `DynamicRange`:
                缩放后的新动态范围。
        """
        assert ratio is not None, "ratio must be specified"
        if zero_domain is None:
            # 对称量化
            assert self.max is not None, "self.max must be specified"
            assert self.min is None, "self.min must be None for data type without zero-point"
            max_value = self.max * ratio
            min_value = None
        else:
            # 非对称量化
            assert self.min is not None, "self.min must be specified"
            assert self.max is not None, "self.max must be specified"
            if is_float_point:
                # 浮点量化：以中心点对称缩放
                centroid_value = (self.min + self.max) / 2
                vmag = (self.max - centroid_value) * ratio
                max_value = centroid_value + vmag
                min_value = centroid_value - vmag
            else:
                # 整数量化：直接缩放
                min_value = self.min * ratio
                max_value = self.max * ratio
            
            # PreScale零点域：确保范围包含0
            if zero_domain == ZeroPointDomain.PreScale:
                max_value = max_value.clamp(min=0)
                min_value = min_value.clamp(max=0)
        return DynamicRange(min=min_value, max=max_value)

    @staticmethod
    def construct(
        tensors: torch.Tensor | list[torch.Tensor],
        /,
        *,
        zero_domain: ZeroPointDomain | None,
        is_float_point: bool,
    ) -> "DynamicRange":
        """从张量构建动态范围的便捷方法。
        
        Args:
            tensors: 输入张量。
            zero_domain: 零点域。
            is_float_point: 是否为浮点量化。
            
        Returns:
            测量得到的动态范围。
        """
        return DynamicRange().measure(tensors, zero_domain=zero_domain, is_float_point=is_float_point)

    @staticmethod
    def _format_m_(
        value: torch.Tensor | float | None,
        *,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        """格式化范围值为指定形状的张量（内部方法）。
        
        Args:
            value: 输入值。
            shape: 目标形状。
            dtype: 目标数据类型。
            device: 目标设备。
            
        Returns:
            格式化后的张量或None。
        """
        if value is None:
            return None
        elif isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.view(-1).to(dtype=dtype, device=device).expand(shape)
            elif value.numel() == shape.numel():
                return value.view(shape).to(dtype=dtype, device=device)
            elif value.shape[1:] == shape[1:] and value.shape[0] == 1:
                return value.to(dtype=dtype, device=device).expand(shape)
            else:
                raise ValueError(f"Invalid value shape: {value.shape}")
        else:
            return torch.full(shape, value, dtype=dtype, device=device)

    def to_dict(self) -> dict[str, tp.Any]:
        """转换为字典表示。
        
        Returns:
            包含min、max和ratio的字典。
        """
        return {"min": self.min, "max": self.max, "ratio": self.ratio}

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any] | None) -> tp.Optional[tp.Self]:
        """从字典创建动态范围。
        
        Args:
            data: 包含min、max和ratio的字典，或None。
            
        Returns:
            DynamicRange实例或None。
        """
        return cls(min=data["min"], max=data["max"], ratio=data["ratio"]) if data is not None else None
