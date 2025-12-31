# -*- coding: utf-8 -*-
"""量化数据模块。

本模块提供了模型量化所需的核心数据结构，包括：
- 量化数据类型定义
- 量化范围计算
- 缩放因子管理
- 量化张量封装

主要导出:
    - QDType: 量化数据类型便捷访问类
    - QuantDataType: 量化数据类型类
    - DynamicRange: 动态范围类
    - LogQuantRange: 对数量化范围类
    - QuantRange: 量化范围类
    - RangeBound: 范围边界类
    - QuantScale: 量化缩放因子类
    - QuantTensor: 量化张量类
"""

from .dtype import QDType, QuantDataType
from .range import DynamicRange, LogQuantRange, QuantRange, RangeBound
from .scale import QuantScale
from .tensor import QuantTensor
