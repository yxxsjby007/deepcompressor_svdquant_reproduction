# -*- coding: utf-8 -*-
"""量化码本模块。

本模块定义了量化过程中使用的码本(Codebook)数据结构。
码本是一个值到二进制编码的映射表，用于非均匀量化和浮点量化。

码本的作用:
    1. 定义量化的可表示值集合
    2. 提供值到编码的映射
    3. 支持将任意浮点数四舍五入到最近的可表示值

应用场景:
    - 浮点量化 (FP4, FP8 等)
    - 非均匀量化
    - 自定义量化格式

主要组件:
    - Codebook: 码本数据类，包含值表和编码表
"""

from dataclasses import dataclass

import torch

from deepcompressor.csrc.load import _C

__all__ = ["Codebook"]


@dataclass
class Codebook:
    """量化码本类。
    
    码本定义了量化格式中所有可表示的值及其对应的二进制编码。
    值按升序排列，便于快速查找最近邻。

    Attributes:
        size (`int`):
            码本中有效值的数量。
        bits (`int`):
            二进制编码的位数。
        values (`torch.FloatTensor`):
            按升序排列的值表。
        codes (`torch.ByteTensor`):
            对应的二进制编码表。
    
    Example:
        >>> # 创建一个简单的2位码本
        >>> maps = [(0.0, 0), (0.5, 1), (1.0, 2), (1.5, 3)]
        >>> codebook = Codebook.construct(maps, bits=2)
        >>> print(codebook.values)  # tensor([0.0, 0.5, 1.0, 1.5])
    """

    size: int               # 码本大小（有效值数量）
    bits: int               # 编码位数
    values: torch.Tensor    # 值表（升序排列）
    codes: torch.Tensor     # 编码表

    def __post_init__(self):
        """初始化后的验证。"""
        assert self.size <= self.values.numel(), "Codebook size is larger than the values size"
        assert self.values.shape == self.codes.shape, "Values and Codes must have the same shape"

    def round(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量四舍五入到码本中最近的值。
        
        使用CUDA加速的最近邻查找算法。

        Args:
            tensor (`torch.Tensor`):
                要量化的输入张量。

        Returns:
            `torch.Tensor`:
                量化后的张量，所有值都是码本中的可表示值。
        """
        dtype = tensor.dtype
        tensor = tensor.to(self.values.dtype).contiguous()
        return _C.round_to_nearest_in_codebook_cuda(tensor, self.values).to(dtype=dtype)

    def to(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> "Codebook":
        """将码本移动到指定设备和数据类型。

        Args:
            device (`torch.device`):
                目标设备。
            dtype (`torch.dtype`):
                目标数据类型（仅应用于values）。

        Returns:
            `Codebook`:
                新的码本对象。
        """
        device = device if device is not None else self.values.device
        dtype = dtype if dtype is not None else self.values.dtype
        return Codebook(
            size=self.size,
            bits=self.bits,
            values=self.values.to(device=device, dtype=dtype),
            codes=self.codes.to(device=device),
        )

    @staticmethod
    def construct(
        maps: list[tuple[float, int]],
        *,
        bits: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "Codebook":
        """从值-编码映射列表构建码本。

        Args:
            maps (`list[tuple[float, int]]`):
                (值, 二进制编码) 元组列表。
            bits (`int`):
                二进制编码的位数。
            device (`torch.device` or str, *optional*, defaults to `"cpu"`):
                码本所在设备。
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                值表的数据类型。

        Returns:
            `Codebook`:
                构建的码本对象。
        """
        if bits > 8:
            raise NotImplementedError("Codebook with more than 8 bits is not supported")
        assert len(maps) <= 2**bits, "Too many (value, code) maps for the code bits"
        size = len(maps)
        # 按值升序排序，便于最近邻查找
        maps.sort(key=lambda x: x[0])
        values = torch.tensor([v[0] for v in maps], device=device, dtype=dtype)
        codes = torch.tensor(
            [v[1] for v in maps],
            dtype=torch.uint8 if bits <= 8 else (torch.int16 if bits < 16 else torch.int32),
            device=device,
        )
        return Codebook(size=size, bits=bits, values=values, codes=codes)

    @staticmethod
    def build_for_float_point(
        *,
        total_bits: int,
        exponent_bits: int,
        signed: bool = True,
        has_subnormal: bool = True,
        has_inf: bool = False,
        has_nan: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "Codebook":
        """为浮点数据类型构建码本。
        
        根据IEEE 754风格的浮点格式参数生成所有可表示值。

        Args:
            total_bits (`int`):
                总位数。
            exponent_bits (`int`):
                指数位数。
            signed (`bool`, *optional*, defaults to `True`):
                是否有符号。
            has_subnormal (`bool`, *optional*, defaults to `True`):
                是否支持次正规数。
            has_inf (`bool`, *optional*, defaults to `False`):
                是否包含无穷大。
            has_nan (`bool`, *optional*, defaults to `False`):
                是否包含NaN。
            device (`torch.device` or str, *optional*, defaults to `"cpu"`):
                码本所在设备。
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                值表的数据类型。

        Returns:
            `Codebook`:
                浮点格式的码本。
        """
        # 计算尾数位数
        mantissa_bits = total_bits - exponent_bits - int(signed)
        assert exponent_bits > 0, "Exponent bits must be positive"
        assert mantissa_bits >= 0, "Mantissa bits must be non-negative"
        has_nan = has_inf or has_nan  # 有inf则必有nan

        # 计算各种掩码和范围
        sign_mask = 1 << (total_bits - 1)  # 符号位掩码
        if mantissa_bits > 0:
            end_evalue = 2**exponent_bits - int(has_inf)
        else:
            end_evalue = 2**exponent_bits - int(has_nan)
        end_mvalue = 2**mantissa_bits
        bias = 2 ** (exponent_bits - 1) - 1  # 指数偏移量
        
        # 生成所有可表示值
        maps, code = [], 0
        for evalue in range(end_evalue):
            for mvalue in range(end_mvalue):
                if evalue == 0 and has_subnormal:
                    # 次正规数: value = (m / 2^mantissa_bits) * 2^(1-bias)
                    value = (mvalue / end_mvalue) * (2 ** (1 - bias))
                else:
                    # 正规数: value = (1 + m / 2^mantissa_bits) * 2^(e-bias)
                    value = (1 + mvalue / end_mvalue) * (2 ** (evalue - bias))
                maps.append((value, code))
                if signed:
                    # 添加负数
                    maps.append((-value, code | sign_mask))
                code += 1
        
        # 处理NaN（移除最后一个编码）
        if mantissa_bits > 0 and not has_inf and has_nan:
            maps = maps[: -(1 + int(signed))]
        return Codebook.construct(maps, bits=total_bits, device=device, dtype=dtype)

    @staticmethod
    def build_for_integer(
        *,
        total_bits: int,
        signed: bool = True,
        magnitude: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "Codebook":
        """为整数数据类型构建码本。

        Args:
            total_bits (`int`):
                总位数。
            signed (`bool`, *optional*, defaults to `True`):
                是否有符号。
            magnitude (`bool`, *optional*, defaults to `False`):
                是否使用幅度表示（符号+绝对值）而非二进制补码。
            device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
                码本所在设备。
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                值表的数据类型。

        Returns:
            `Codebook`:
                整数格式的码本。
        """
        if signed:
            end_value = 2 ** (total_bits - 1)  # 正数最大值+1
            min_value = -end_value + int(magnitude)  # 负数最小值
        else:
            end_value = 2**total_bits
            min_value = 0
        
        maps = []
        for value in range(min_value, end_value):
            if value >= 0:
                code = value
            elif magnitude:
                # 幅度表示: 负数编码 = 2^(n-1) + |value|
                code = end_value - value
            else:
                # 二进制补码: 负数编码 = 2^n + value
                code = end_value + end_value + value
            maps.append((value, code))
        return Codebook.construct(maps, bits=total_bits, device=device, dtype=dtype)
