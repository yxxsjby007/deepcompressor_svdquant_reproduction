# -*- coding: utf-8 -*-
"""量化数据类型模块。

本模块定义了用于模型量化的自定义数据类型系统。支持多种量化格式，包括：
- 整数量化 (INT4, INT8 等)
- 浮点量化 (FP4, FP8 等)
- 自定义码本量化

量化数据类型是模型压缩的核心概念，它决定了：
1. 量化后的数值范围
2. 量化精度和误差特性
3. 与硬件加速器的兼容性

主要组件:
    - QuantDataType: 量化数据类型类，支持整数和浮点格式
    - QDType: 便捷访问类，通过属性名快速获取预定义的数据类型

支持的数据类型格式:
    - 有符号/无符号整数: sint4, uint4, sint8, uint8 等
    - 有符号/无符号浮点: sfp4_e2m1, sfp8_e4m3 等
    - 幅度表示整数: smag4, smag8 等
    - 指数表示: sexp4, sexp8 等

Example:
    >>> # 获取4位有符号整数类型
    >>> int4 = QuantDataType.from_str("sint4")
    >>> print(int4.max_value)  # 7
    >>> print(int4.min_value)  # -8
    >>> 
    >>> # 使用便捷类
    >>> fp4 = QDType.sfp4_e2m1
"""

import typing as tp

import torch

from .codebook import Codebook

__all__ = ["QuantDataType", "QDType"]


class QuantDataType:
    """量化数据类型类。
    
    该类定义了量化过程中使用的数据类型，支持整数和浮点两种基本格式。
    每种格式都可以配置位宽、符号、指数位等参数。
    
    数据类型分类:
        1. 整数类型 (exponent_bits=0):
           - 标准整数: 使用二进制补码表示负数
           - 幅度整数: 使用符号位+幅度表示
           
        2. 浮点类型 (exponent_bits>0):
           - 标准浮点: 符号位 + 指数位 + 尾数位
           - 支持次正规数(subnormal)、无穷大(inf)、非数(NaN)
    
    Attributes:
        name: 数据类型的字符串名称
        signed: 是否为有符号类型
        total_bits: 总位宽
        exponent_bits: 指数位数（浮点类型）
        mantissa_bits: 尾数位数（浮点类型）
        max_value: 可表示的最大值
        min_value: 可表示的最小值
    
    Example:
        >>> # 创建4位有符号整数类型
        >>> int4 = QuantDataType(4, signed=True)
        >>> print(int4.max_value)  # 7
        >>> 
        >>> # 创建FP8 E4M3格式
        >>> fp8 = QuantDataType(8, signed=True, exponent_bits=4)
    """

    # 类变量：已注册的数据类型缓存，避免重复创建相同类型
    _registered: tp.ClassVar[dict[str, "QuantDataType"]] = {}

    def __init__(
        self,
        total_bits: int,
        *,
        signed: bool = True,
        exponent_bits: int = 0,
        has_subnormal: bool = True,
        has_nan: bool = False,
        has_inf: bool = False,
        magnitude: bool = False,
        codebook: Codebook | None = None,
        codebook_name: str = "",
    ):
        """初始化量化数据类型。

        Args:
            total_bits (`int`):
                总位宽，必须大于0。例如：4表示4位量化，8表示8位量化。
            signed (`bool`, *optional*, defaults to `True`):
                是否为有符号类型。True表示可以表示负数。
            exponent_bits (`int`, *optional*, defaults to `0`):
                指数位数。0表示整数类型，>0表示浮点类型。
                例如：FP8 E4M3格式中exponent_bits=4。
            has_subnormal (`bool`, *optional*, defaults to `True`):
                是否支持次正规数（仅浮点类型有效）。
                次正规数可以表示更接近0的小数值。
            has_nan (`bool`, *optional*, defaults to `False`):
                是否保留NaN编码（仅浮点类型有效）。
            has_inf (`bool`, *optional*, defaults to `False`):
                是否保留无穷大编码（仅浮点类型有效）。
            magnitude (`bool`, *optional*, defaults to `False`):
                是否使用幅度表示（仅整数类型有效）。
                幅度表示使用符号位+绝对值，而非二进制补码。
            codebook (`Codebook` or `None`, *optional*, defaults to `None`):
                自定义码本。用于非标准量化格式。
            codebook_name (`str`, *optional*, defaults to `""`):
                码本名称。如果提供了codebook，则必须指定此参数。
        """
        self.__signed = signed
        # region 设置位宽参数
        # total_bits: 总位宽 = 符号位(如果有) + 指数位 + 尾数位
        self.__total_bits = total_bits
        self.__exponent_bits = exponent_bits
        assert self.__total_bits > 0, "Total bits must be greater than 0."
        assert self.__exponent_bits >= 0, "Exponent bits must be non-negative."
        # 计算尾数位数: 总位宽 - 指数位 - 符号位(1或0)
        self.__mantissa_bits = self.__total_bits - self.__exponent_bits - int(self.__signed)
        # endregion
        
        # region 设置数据类型属性
        if self.__exponent_bits > 0:
            # 浮点数据类型的属性设置
            self.__has_subnormal = has_subnormal  # 是否支持次正规数
            self.__has_inf = has_inf              # 是否支持无穷大
            self.__has_nan = has_inf or has_nan   # 是否支持NaN（有inf则必有nan）
            self.__magnitude = True               # 浮点数总是使用幅度表示
            if self.__mantissa_bits == 0:
                # 纯指数格式（无尾数位）的特殊限制
                assert not self.__has_inf, "Inf is not supported for exponent-only floating-point data type."
                if self.__exponent_bits == 1:
                    assert not self.__has_nan, "NaN is not supported for 1-bit exponent-only floating-point data type."
        else:
            # 整数数据类型的属性设置
            self.__has_subnormal = False  # 整数无次正规数概念
            self.__has_inf = False        # 整数无无穷大
            self.__has_nan = False        # 整数无NaN
            self.__magnitude = magnitude  # 是否使用幅度表示（符号+绝对值）
        # endregion
        
        # region 设置码本（用于自定义量化格式）
        if codebook is not None:
            assert self.is_float_point, "Codebook is only supported for floating-point data type."
            self.__codebook = codebook
            assert codebook_name, "Codebook name must be specified."
            self.__codebook_name = codebook_name
            assert self.max_value >= 0, "Max value must be non-negative."
            self.__name = self.__codebook_name
            # 检查是否已注册相同名称的类型，确保一致性
            if self.__name not in QuantDataType._registered:
                QuantDataType._registered[self.__name] = self
            else:
                _registered = QuantDataType._registered[self.__name]
                assert _registered.total_bits == self.total_bits, "Total bits must be the same as the registered one."
                assert _registered.exponent_bits == self.exponent_bits, (
                    "Exponent bits must be the same as the registered one."
                )
                assert _registered.signed == self.signed, "Signed must be the same as the registered one."
                assert _registered.has_subnormal == self.has_subnormal, (
                    "Subnormal must be the same as the registered one."
                )
                assert _registered.has_inf == self.has_inf, "Inf must be the same as the registered one."
                assert _registered.has_nan == self.has_nan, "NaN must be the same as the registered one."
                assert _registered.magnitude == self.magnitude, "Magnitude must be the same as the registered one."
                assert _registered.__codebook is not None, "Codebook must be the same as the registered one."
                assert torch.allclose(_registered.__codebook.values, self.__codebook.values), (
                    "Codebook values must be the same as the registered one."
                )
        else:
            self.__codebook = None
            self.__codebook_name = ""
            self.__name = self._build_default_name()  # 根据参数自动生成名称
            if self.__name not in QuantDataType._registered:
                QuantDataType._registered[self.__name] = self
        # endregion
        
        # region 初始化码本缓存（按设备和数据类型缓存）
        self.__codebooks: dict[tuple[torch.device, torch.dtype], Codebook] = {}
        # endregion

    # region 属性访问器
    @property
    def name(self) -> str:
        """数据类型的字符串名称。"""
        return self.__name

    @property
    def codebook_name(self) -> str:
        """码本名称（如果使用自定义码本）。"""
        return self.__codebook_name

    @property
    def signed(self) -> bool:
        """是否为有符号类型。"""
        return self.__signed

    @property
    def unsigned(self) -> bool:
        """是否为无符号类型。"""
        return not self.__signed

    @property
    def total_bits(self) -> int:
        """总位宽。"""
        return self.__total_bits

    @property
    def exponent_bits(self) -> int:
        """指数位数（浮点类型）。整数类型返回0。"""
        return self.__exponent_bits

    @property
    def mantissa_bits(self) -> int:
        """尾数位数（浮点类型）。整数类型返回数值位数。"""
        return self.__mantissa_bits

    @property
    def has_subnormal(self) -> bool:
        """是否支持次正规数（仅浮点类型）。"""
        return self.__has_subnormal

    @property
    def has_inf(self) -> bool:
        """是否支持无穷大（仅浮点类型）。"""
        return self.__has_inf

    @property
    def has_nan(self) -> bool:
        """是否支持NaN（仅浮点类型）。"""
        return self.__has_nan

    @property
    def magnitude(self) -> bool:
        """是否使用幅度表示（符号位+绝对值）。"""
        return self.__magnitude

    @property
    def is_float_point(self) -> bool:
        """是否为浮点类型。"""
        return self.exponent_bits > 0

    @property
    def is_integer(self) -> bool:
        """是否为整数类型。"""
        return self.exponent_bits == 0

    @property
    def is_exponent(self) -> bool:
        """是否为纯指数浮点类型（无尾数位且无次正规数）。"""
        return self.exponent_bits > 0 and self.mantissa_bits == 0 and not self.has_subnormal

    @property
    def exponent_mask(self) -> int:
        """指数位的位掩码。用于从二进制表示中提取指数部分。"""
        return ((1 << self.exponent_bits) - 1) << self.mantissa_bits

    @property
    def mantissa_mask(self) -> int:
        """尾数位的位掩码。用于从二进制表示中提取尾数部分。"""
        return (1 << self.mantissa_bits) - 1

    @property
    def _end_mantissa(self) -> int:
        """尾数的最大值+1（用于内部计算）。"""
        return 2**self.mantissa_bits

    @property
    def _end_exponent(self) -> int:
        """有效指数值的最大值+1（用于内部计算）。"""
        if self.mantissa_bits > 0:
            return 2**self.exponent_bits - int(self.has_inf)
        else:
            return 2**self.exponent_bits - int(self.has_nan)

    @property
    def exponent_bias(self) -> int:
        """指数偏移量（bias）。
        
        浮点数的实际指数 = 存储的指数值 - bias
        例如：FP8 E4M3的bias = 2^(4-1) - 1 = 7
        """
        if self.is_float_point:
            return 2 ** (self.exponent_bits - 1) - 1
        else:
            return 0

    @property
    def max_exponent_value(self) -> int:
        """最大指数值（实际值，已减去bias）。"""
        if self.is_float_point:
            return self._end_exponent - 1 - self.exponent_bias
        else:
            return self.total_bits - 1 - int(self.signed)

    @property
    def min_exponent_value(self) -> int:
        """最小指数值（实际值，已减去bias）。"""
        if self.is_float_point:
            return int(self.has_subnormal) - self.exponent_bias
        else:
            return 0

    @property
    def max_positive_normal_value(self) -> float:
        """最大正正规数值。
        
        对于浮点类型：(2 - 2^(-mantissa_bits)) * 2^max_exponent
        对于整数类型：2^mantissa_bits - 1
        """
        if self.is_float_point:
            if self.mantissa_bits > 0 and not self.has_inf and self.has_nan:
                base_value = 2 - 2 / self._end_mantissa
            else:
                base_value = 2 - 1 / self._end_mantissa
            return base_value * 2**self.max_exponent_value
        else:
            return self._end_mantissa - 1

    @property
    def min_positive_normal_value(self) -> float:
        """最小正正规数值。"""
        return 2**self.min_exponent_value

    @property
    def max_positive_subnormal(self) -> float:
        """最大正次正规数值。"""
        if self.is_float_point and self.has_subnormal and self.mantissa_bits > 0:
            b = 1 - 1 / self._end_mantissa
            e = 1 - self.exponent_bias
            return b * 2**e
        else:
            return 0

    @property
    def min_positive_subnormal(self) -> float:
        """最小正次正规数值（最接近0的正数）。"""
        if self.is_float_point and self.has_subnormal and self.mantissa_bits > 0:
            b = 1 / self._end_mantissa
            e = 1 - self.exponent_bias
            return b * 2**e
        else:
            return 0

    @property
    def max_value(self) -> float:
        """该数据类型可表示的最大值。"""
        return self.max_positive_normal_value if self.__codebook is None else self.__codebook.values[-1].item()

    @property
    def min_value(self) -> float:
        """该数据类型可表示的最小值。
        
        对于有符号类型：
            - 幅度表示: -max_value
            - 补码表示: -max_value - 1
        对于无符号类型: 0
        """
        if self.__codebook is not None:
            return self.__codebook.values[0].item()
        if self.signed:
            if self.magnitude:
                return -self.max_value
            else:
                return -self.max_value - 1
        else:
            return 0

    # endregion

    def to_unsigned(self) -> "QuantDataType":
        """获取该数据类型的无符号版本。

        Returns:
            `QuantDataType`:
                无符号版本的数据类型。
                
        Example:
            >>> sint4 = QuantDataType.from_str("sint4")
            >>> uint4 = sint4.to_unsigned()
            >>> print(uint4.name)  # "uint4"
        """
        return QuantDataType.from_str("u" + self.name[1:])

    def get_codebook(self, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> Codebook:
        """获取该数据类型的码本。
        
        码本包含了该数据类型所有可能的量化值，用于将浮点数映射到最近的量化值。
        码本会按(device, dtype)进行缓存，避免重复创建。

        Args:
            device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
                码本所在的设备。
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                码本的数据类型。

        Returns:
            `Codebook`:
                指定设备和数据类型的码本。
        """
        device = torch.device("cpu") if device is None else torch.device(device)
        key = (device, dtype)
        if key not in self.__codebooks:
            if self.__codebook is not None:
                self.__codebooks[key] = self.__codebook.to(device=device, dtype=dtype)
            else:
                self.__codebook = self._build_codebook(device=device, dtype=dtype)
                self.__codebooks[key] = self.__codebook
        return self.__codebooks[key]

    def round(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量四舍五入到最近的量化值。
        
        对于整数类型，使用标准的四舍五入。
        对于浮点类型，使用码本查找最近的可表示值。

        Args:
            tensor (`torch.Tensor`):
                要量化的张量。

        Returns:
            `torch.Tensor`:
                量化后的张量。
        """
        if self.is_integer:
            return tensor.round()
        else:
            return self.get_codebook(device=tensor.device).round(tensor)

    @classmethod
    def from_str(cls, s: str, /) -> "QuantDataType":
        """从字符串创建量化数据类型。
        
        支持的格式:
            - 整数: "sint4", "uint8", "smag4" 等
            - 浮点: "sfp4_e2m1", "sfp8_e4m3_inf" 等
            - 指数: "sexp4", "uexp3" 等
        
        Args:
            s: 数据类型的字符串表示。
            
        Returns:
            对应的QuantDataType实例。
            
        Example:
            >>> int4 = QuantDataType.from_str("sint4")
            >>> fp8 = QuantDataType.from_str("sfp8_e4m3")
        """
        if s not in cls._registered:
            cls._registered[s] = cls._default_from_str(s)
        return cls._registered[s]

    def _build_codebook(self, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> Codebook:
        """构建该数据类型的码本（内部方法）。"""
        if self.is_float_point:
            return Codebook.build_for_float_point(
                total_bits=self.total_bits,
                exponent_bits=self.exponent_bits,
                signed=self.signed,
                has_subnormal=self.has_subnormal,
                has_inf=self.has_inf,
                has_nan=self.has_nan,
                device=device,
                dtype=dtype,
            )
        else:
            return Codebook.build_for_integer(
                total_bits=self.total_bits, signed=self.signed, magnitude=self.magnitude, device=device, dtype=dtype
            )

    def _build_default_name(self) -> str:
        """根据数据类型参数构建默认名称（内部方法）。
        
        命名规则:
            - 前缀: 's'(有符号) 或 'u'(无符号)
            - 整数: 'int' + 位宽 (如 sint4) 或 'mag' + 位宽 (如 smag4)
            - 浮点: 'fp'/'fn' + 位宽 + '_e' + 指数位 + 'm' + 尾数位 + 后缀
            - 指数: 'exp' + 位宽 + 后缀
        """
        s = "s" if self.signed else "u"
        if self.is_float_point:
            if self.has_subnormal or self.mantissa_bits > 0:
                s += "fp" if self.has_subnormal else "fn"  # fp=有次正规数, fn=无次正规数
                s += f"{self.total_bits}_e{self.exponent_bits}m{self.mantissa_bits}"
                s += "_inf" if self.has_inf else ("_nan" if self.has_nan else "_all")
            else:
                # 纯指数格式
                assert not self.has_subnormal, "Subnormal is not supported for exponent-only floating-point data type."
                assert not self.has_inf, "Inf is not supported for exponent-only floating-point data type."
                s += f"exp{self.exponent_bits}"
                s += "_nan" if self.has_nan else "_all"
        else:
            s += "mag" if self.magnitude else "int"  # mag=幅度表示, int=补码表示
            s += f"{self.total_bits}"
        return s

    @staticmethod
    def _default_from_str(s: str, /) -> "QuantDataType":
        """从字符串解析并创建数据类型（内部方法）。
        
        解析格式:
            - sint4, uint8: 标准整数
            - smag4: 幅度表示整数
            - sexp4: 纯指数浮点
            - sfp4_e2m1: 标准浮点 (有次正规数)
            - sfn4_e2m1: ���次正规数浮点
        """
        s = s.strip().lower()
        signed = s[0] == "s"  # 首字符判断符号
        s = s[1:]
        if s.startswith("int"):
            # 标准整数: sint4 -> int4
            return QuantDataType(int(s[3:]), signed=signed)
        elif s.startswith("mag"):
            # 幅度整数: smag4 -> mag4
            return QuantDataType(int(s[3:]), signed=signed, magnitude=True)
        elif s.startswith("exp"):
            # 纯指数浮点: sexp4 -> exp4
            ss = s.split("_")
            total_bits = int(ss[0][3:])
            if len(ss) >= 2:
                has_nan = ss[1] == "nan"
            else:
                has_nan = False
            return QuantDataType(
                total_bits=total_bits,
                signed=signed,
                exponent_bits=total_bits - int(signed),
                has_subnormal=False,
                has_nan=has_nan,
            )
        elif s.startswith("f"):
            # 浮点类型: sfp4_e2m1 或 sfn4_e2m1
            ss = s.split("_")
            has_subnormal = s[1] == "p"  # fp=有次正规数, fn=无
            total_bits = int(ss[0][2:])
            exponent_bits = int(ss[1][1 : ss[1].find("m")])
            if len(ss) >= 3:
                has_inf = ss[2] == "inf"
                has_nan = has_inf or (ss[2] == "nan")
            else:
                has_inf, has_nan = False, False
            return QuantDataType(
                total_bits=total_bits,
                signed=signed,
                exponent_bits=exponent_bits,
                has_subnormal=has_subnormal,
                has_inf=has_inf,
                has_nan=has_nan,
            )
        else:
            raise ValueError(f"Unknown QuantDataType {s}")

    def __str__(self) -> str:
        """返回数据类型的字符串表示。"""
        return self.__name

    def __repr__(self) -> str:
        """返回数据类型的字��串表示。"""
        return self.__name

    def __eq__(self, value: object) -> bool:
        """判断两个数据类型是否相等（基于名称）。"""
        if not isinstance(value, QuantDataType):
            return False
        return self.name == value.name

    def __hash__(self) -> int:
        """返回数据类型的哈希值（基于名称）。"""
        return hash(self.name)


class _QDTypeMeta(type):
    """QDType的元类，实现属性访问时自动创建QuantDataType。
    
    通过元类的__getattr__方法，可以实现类似 QDType.sint4 的便捷访问方式，
    自动将属性名转换为对应的QuantDataType实例。
    """
    
    def __getattr__(cls, __name: str) -> tp.Any:
        if __name.startswith("_"):
            # 私有属性使用默认行为
            return getattr(super(), __name)
        else:
            # 公共属性名作为数据类型名称解析
            return QuantDataType.from_str(__name)


class QDType(metaclass=_QDTypeMeta):
    """量化数据类型的便捷访问类。
    
    通过属性访问的方式快速获取预定义的量化数据类型，
    无需手动调用 QuantDataType.from_str()。
    
    支持的数据类型示例:
        - 整数类型: QDType.sint4, QDType.uint8, QDType.sint8
        - 幅度整数: QDType.smag4, QDType.smag8
        - 浮点类型: QDType.sfp4_e2m1, QDType.sfp8_e4m3
        - 指数类型: QDType.sexp4, QDType.sexp8
    
    Example:
        >>> # 获取4位有符号整数类型
        >>> int4 = QDType.sint4
        >>> print(int4.max_value)  # 7
        >>> print(int4.min_value)  # -8
        >>> 
        >>> # 获取FP8 E4M3类型
        >>> fp8 = QDType.sfp8_e4m3
        >>> print(fp8.is_float_point)  # True
        >>> 
        >>> # 获取4位无符号整数类型
        >>> uint4 = QDType.uint4
        >>> print(uint4.signed)  # False
    
    Note:
        属性名必须符合QuantDataType的命名规范，否则会抛出ValueError。
    """

    pass
