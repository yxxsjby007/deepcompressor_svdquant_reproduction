# -*- coding: utf-8 -*-
"""GPTQ量化核心模块。

本模块实现了GPTQ (Generative Pre-trained Transformer Quantization) 算法。
GPTQ是一种基于Hessian矩阵的逐层量化优化方法，可以显著提高低位宽量化的精度。

GPTQ算法原理:
    1. 使用校准数据计算Hessian矩阵 H = X^T * X
    2. 按重要性对列进行排序
    3. 逐列量化，同时补偿量化误差到未量化的列
    4. 误差补偿公式: W[:, j+1:] -= error * H_inv[j, j+1:] / H_inv[j, j]

优点:
    - 相比RTN，在低位宽（如4位）下精度损失更小
    - 只需要少量校准数据
    - 支持分块处理，内存效率高

缺点:
    - 需要校准数据
    - 计算复杂度较高
    - 需要计算和存储Hessian矩阵

参考文献:
    Frantar, E., et al. (2022). GPTQ: Accurate Post-Training Quantization 
    for Generative Pre-trained Transformers.

主要组件:
    - QuantGptqConfig: GPTQ配置类
    - QuantGptqKernel: GPTQ量化核心类
    - gptq_quantize: GPTQ量化函数
"""

import gc
import math
from dataclasses import dataclass

import torch
from omniconfig import configclass

from ...data.cache import TensorCache
from ...data.dtype import QuantDataType
from ...data.range import QuantRange, RangeBound
from ...data.zero import ZeroPointDomain
from ...utils import tools
from ...utils.common import num2str
from ..config.kernel import BaseQuantKernel, BaseQuantKernelConfig
from ..impl.simple import simple_quantize

__all__ = ["gptq_quantize"]


@configclass
@dataclass
class QuantGptqConfig(BaseQuantKernelConfig):
    """GPTQ量化配置类。
    
    配置GPTQ算法的各种超参数。

    Args:
        damp_percentage (`float`, *optional*, defaults to `0.01`):
            阻尼百分比。用于数值稳定性，防止Hessian矩阵奇异。
            实际阻尼值 = damp_percentage * mean(diag(H))
        block_size (`int`, *optional*, defaults to `128`):
            GPTQ的块大小。每次处理block_size列，然后更新剩余列。
            较大的块大小可能提高精度但增加内存使用。
        num_inv_tries (`int`, *optional*, defaults to `200`):
            Cholesky分解的最大尝试次数。如果分解失败，会增加阻尼重试。
        hessian_block_size (`int`, *optional*, defaults to `-1`):
            计算Hessian矩阵时的块大小。-1表示不分块。
            用于处理大规模校准数据时的内存优化。
            
    Example:
        >>> config = QuantGptqConfig(
        ...     damp_percentage=0.01,
        ...     block_size=128,
        ... )
        >>> kernel = config.build()
    """

    damp_percentage: float = 0.01   # 阻尼百分比
    block_size: int = 128           # 块大小
    num_inv_tries: int = 200        # 最大求逆尝试次数
    hessian_block_size: int = -1    # Hessian计算块大小

    @property
    def name(self) -> str:
        """配置名称。"""
        return "GPTQ"

    def build(self) -> "QuantGptqKernel":
        """构建GPTQ量化核心。
        
        Returns:
            QuantGptqKernel实例。
        """
        return QuantGptqKernel(self)

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """生成配置的目录名称。

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                目录名称前缀。

        Returns:
            `list[str]`:
                目录名称列表。
        """
        name = f"gptq.d{num2str(self.damp_percentage)}.b{num2str(self.block_size)}"
        return [f"{prefix}.{name}" if prefix else name]


class QuantGptqKernel(BaseQuantKernel):
    """GPTQ量化核心类。
    
    实现了基于Hessian矩阵的GPTQ量化算法。
    
    Attributes:
        config: GPTQ配置对象。
        
    Example:
        >>> config = QuantGptqConfig()
        >>> kernel = QuantGptqKernel(config)
        >>> qtensor = kernel.quantize(
        ...     tensor=weight,
        ...     view_shape=view_shape,
        ...     quant_dtype=QDType.sint4,
        ...     zero_domain=None,
        ...     scale=scale,
        ...     zero=zero,
        ...     inputs=input_cache,  # 校准数据
        ... )
    """
    
    def __init__(self, config: "QuantGptqConfig"):
        """初始化GPTQ量化核心。
        
        Args:
            config: GPTQ配置对象。
        """
        self.config = config

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        view_shape: torch.Size,
        quant_dtype: QuantDataType,
        zero_domain: ZeroPointDomain | None,
        scale: torch.Tensor,
        zero: torch.Tensor,
        inputs: TensorCache,
        quant_range: QuantRange | None = None,
        range_bound: RangeBound | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """执行GPTQ量化。

        Args:
            tensor (`torch.Tensor`):
                要量化的张量（通常是权重矩阵）。
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
            inputs (`TensorCache`):
                输入激活值的缓存，用于计算Hessian矩阵。
            quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                量化范围约束。
            range_bound (`RangeBound` or `None`, *optional*, defaults to `None`):
                范围边界约束。
            **kwargs: 其他关键字参数。

        Returns:
            `torch.Tensor`:
                量化后的张量，形状为 ``view_shape``。
        """
        assert not tensor.requires_grad, "tensor must not require gradient."
        assert not scale.data.requires_grad, "scale must not require gradient."
        assert not zero.data.requires_grad, "zero must not require gradient."
        return gptq_quantize(
            tensor,
            view_shape=view_shape,
            quant_dtype=quant_dtype,
            zero_domain=zero_domain,
            scale=scale,
            zero=zero,
            gptq_config=self.config,
            inputs=inputs,
            quant_range=quant_range,
            range_bound=range_bound,
        )


@torch.no_grad()
def gptq_quantize(  # noqa: C901
    tensor: torch.Tensor,
    *,
    view_shape: torch.Size,
    quant_dtype: QuantDataType,
    zero_domain: ZeroPointDomain | None,
    scale: torch.Tensor,
    zero: torch.Tensor,
    gptq_config: QuantGptqConfig,
    inputs: TensorCache,
    quant_range: QuantRange | None = None,
    range_bound: RangeBound | None = None,
) -> torch.Tensor:
    """使用GPTQ算法量化张量。
    
    GPTQ通过最小化量化误差的二次形式来优化量化结果：
        min ||W - Q||_H^2 = min (W - Q)^T H (W - Q)
    
    其中H是Hessian矩阵，由输入激活计算得到。

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
        gptq_config (`QuantGptqConfig`):
            GPTQ配置。
        inputs (`TensorCache`):
            输入激活值缓存。
        quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
            量化范围约束。
        range_bound (`RangeBound` or `None`, *optional*, defaults to `None`):
            范围边界约束。

    Returns:
        `torch.Tensor`:
            量化后的张量，形状为 ``view_shape``。
    """
    view_tensor = tensor.view(view_shape)
    view_shape = view_tensor.shape  # 移除view_shape中的-1
    
    # region 步骤1: 重塑张量为 (#g0 * gs0, #g1 * #g2 * ... * gs1 * gs2, ...)
    len_view_shape = len(view_shape)
    # view_tensor: (#g0, gs0, #g1, gs1, #g2, gs2, ...) -> (#g0, gs0, #g1, #g2, ..., gs1, gs2, ...)
    reshaped_tensor = view_tensor.permute(0, 1, *range(2, len_view_shape, 2), *range(3, len_view_shape, 2))
    # reshaped_tensor: (#g0 * gs0, #g1 * #g2 * ... * gs1 * gs2 * ...)
    reshaped_tensor = reshaped_tensor.reshape(view_shape[0] * view_shape[1], -1)
    
    # 计算分组信息
    num_row_groups, num_column_groups = view_shape[0], view_shape[2::2].numel()
    row_group_size, column_group_size = view_shape[1], view_shape[3::2].numel()
    num_rows, num_columns = reshaped_tensor.shape
    
    # 重塑缩放因子和零点
    reshaped_scale = scale.view(num_row_groups, 1, num_column_groups)
    zero_is_number = isinstance(zero, (int, float)) or zero.numel() == 1
    reshaped_zero = zero if zero_is_number else zero.view(num_row_groups, 1, num_column_groups)
    # endregion
    
    # region 步骤2: 计算Hessian矩阵 H = X^T * X
    hessian = torch.zeros((num_columns, num_columns), device=view_tensor.device, dtype=view_tensor.dtype)
    for x in inputs.data:
        x: torch.Tensor = inputs.reshape(x.view(-1, *x.shape[inputs.channels_dim :]))
        if gptq_config.hessian_block_size > 0 and x.shape[0] > gptq_config.hessian_block_size:
            # 分块计算Hessian以节省内存
            for b in range(0, x.shape[0], gptq_config.hessian_block_size):
                _x = x[b : min(b + gptq_config.hessian_block_size, x.shape[0])]
                _x = math.sqrt(2 / inputs.num_samples) * _x.to(device=view_tensor.device, dtype=view_tensor.dtype)
                hessian += torch.matmul(_x.t(), _x)
        else:
            x = math.sqrt(2 / inputs.num_samples) * x.to(device=view_tensor.device, dtype=view_tensor.dtype)
            hessian += torch.matmul(x.t(), x)
    
    # 处理死神经元（对角线为0的列）
    dead = hessian.diagonal() == 0
    hessian[dead, dead] = 1
    reshaped_tensor[:, dead] = 0
    del x, inputs, dead
    gc.collect()
    torch.cuda.empty_cache()
    # endregion
    
    # region 步骤3: 按重要性排序列（对角线元素越大越重要）
    importance = torch.diag(hessian)  # (#g1 * #g2 * ... * gs1 * gs2 * ..., )
    permute = torch.argsort(importance, descending=True)
    hessian = hessian[permute][:, permute]
    reshaped_tensor = reshaped_tensor[:, permute]
    inverse_permute = torch.argsort(permute)
    del importance
    # endregion
    
    # region 步骤4: 应用阻尼以避免数值不稳定
    hessian_diag = hessian.diagonal()
    hessian_diag_mean = hessian_diag.mean()
    hessian_diag += gptq_config.damp_percentage * hessian_diag_mean
    # endregion
    
    # region 步骤5: 计算Hessian矩阵的逆（使用Cholesky分解）
    stable_inv, num_inv_tries = False, 0
    while (not stable_inv) and num_inv_tries < gptq_config.num_inv_tries:
        num_inv_tries += 1
        try:
            # Cholesky分解: H = L * L^T
            hessian_inv = torch.linalg.cholesky(hessian)
            # 计算逆: H^{-1} = (L^{-1})^T * L^{-1}
            hessian_inv = torch.cholesky_inverse(hessian_inv)
            # 再次Cholesky分解得到上三角形式
            hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)
        except RuntimeError:
            # 如果分解失败，增加阻尼重试
            hessian_diag += (gptq_config.damp_percentage * 0.1) * hessian_diag_mean
            continue
        stable_inv = True
    
    if num_inv_tries > 1:
        logger = tools.logging.getLogger(f"{__name__}.GPTQ")
        logger.debug("        - Hessian is not stable %s %d tries.", "until" if stable_inv else "after", num_inv_tries)
    
    assert not hessian_inv.isinf().any(), "Inverse of Hessian matrix contains Inf."
    assert not hessian_inv.isnan().any(), "Inverse of Hessian matrix contains NaN."
    del hessian, hessian_diag, hessian_diag_mean, num_inv_tries
    # endregion
    
    # region 步骤6: 逐块量化张量
    qtensor = torch.zeros_like(reshaped_tensor)
    for c_start in range(0, num_columns, gptq_config.block_size):
        c_end = min(c_start + gptq_config.block_size, num_columns)
        block_tensor = reshaped_tensor[:, c_start:c_end].clone()
        block_qtensor = qtensor[:, c_start:c_end]
        block_hessian_inv = hessian_inv[c_start:c_end, c_start:c_end]
        block_error = torch.zeros_like(block_tensor)
        
        # 逐列量化
        for _c in range(c_end - c_start):
            c = c_start + _c
            column = block_tensor[:, _c]  # (#g0 * gs0, )
            pos_diag = block_hessian_inv[_c, _c]
            
            # 获取当前列的缩放因子和零点
            column_group_index = permute[c] // column_group_size
            column_scale = reshaped_scale[:, :, column_group_index]  # (#g0, 1)
            column_zero = reshaped_zero if zero_is_number else reshaped_zero[:, :, column_group_index]
            
            # 量化当前列
            qcolumn = column.view(num_row_groups, row_group_size).clone()  # (#g0, gs0)
            if range_bound is not None and range_bound.is_set():
                qcolumn = qcolumn.clamp_(min=range_bound.min, max=range_bound.max)
            if zero_domain == ZeroPointDomain.PostScale:
                qcolumn = qcolumn.add_(column_zero)
            qcolumn = qcolumn.div_(column_scale)
            if zero_domain == ZeroPointDomain.PreScale:
                qcolumn = qcolumn.add_(column_zero)
            qcolumn = simple_quantize(
                qcolumn, quant_dtype=quant_dtype, has_zero_point=zero_domain is not None, quant_range=quant_range
            )
            block_qtensor[:, _c] = qcolumn.view(-1)  # 保存量化后的列
            
            # 反量化以计算误差
            if zero_domain == ZeroPointDomain.PreScale:
                qcolumn = qcolumn.sub_(column_zero)
            qcolumn = qcolumn.mul_(column_scale)
            if zero_domain == ZeroPointDomain.PostScale:
                qcolumn = qcolumn.sub_(column_zero)
            
            # 计算量化误差并补偿到块内剩余列
            column_error = column.sub_(qcolumn.view(column.shape)).div_(pos_diag)
            block_error[:, _c] = column_error.view(-1)
            block_tensor[:, _c:] -= column_error.view(-1, 1).matmul(block_hessian_inv[_c, _c:].view(1, -1))
        
        # 将块误差补偿到剩余列
        reshaped_tensor[:, c_end:] -= block_error.matmul(hessian_inv[c_start:c_end, c_end:])
    
    # 恢复原始列顺序
    qtensor = qtensor[:, inverse_permute]
    # endregion
    
    # region 步骤7: 重塑张量回原始形状 (#g0, gs0, #g1, gs1, #g2, gs2, ...)
    _view_shape = view_shape[:2] + view_shape[2::2] + view_shape[3::2]
    # qtensor: (#g0 * gs0, #g1 * #g2 * ... * gs1 * gs2, ...) -> (#g0, gs0, #g1, #g2, ..., gs1, gs2, ...)
    qtensor = qtensor.reshape(_view_shape)
    # qtensor: (#g0, gs0, #g1, #g2, ..., gs1, gs2, ...) -> (#g0, gs0, #g1, gs1, #g2, gs2, ...)
    permute_dims = [0, 1]
    for i in range(1, len_view_shape // 2):
        permute_dims.append(1 + i)
        permute_dims.append(len_view_shape // 2 + i)
    qtensor = qtensor.permute(*permute_dims).reshape(view_shape).contiguous()
    # endregion
    
    assert not qtensor.isnan().any(), "GPTQ Quantized tensor contains NaN."
    assert not qtensor.isinf().any(), "GPTQ Quantized tensor contains Inf."
    return qtensor
