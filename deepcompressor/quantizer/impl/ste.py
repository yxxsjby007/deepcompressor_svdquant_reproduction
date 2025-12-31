# -*- coding: utf-8 -*-
"""直通估计器(STE)模块。

本模块实现了直通估计器(Straight-Through Estimator)，用于量化感知训练。
STE允许在前向传播中执行不可微的量化操作，同时在反向传播中直接传递梯度。

背景:
    量化操作（如四舍五入）是不可微的，其梯度几乎处处为零。
    这使得标准的反向传播无法用于训练量化模型。
    STE通过在反向传播时"假装"量化操作是恒等函数来解决这个问题。

数学表示:
    前向: y = f(x)  (f是量化函数，如round)
    反向: ∂L/∂x = ∂L/∂y  (直接传递梯度，忽略f的导数)

应用场景:
    - 量化感知训练(QAT)
    - 可微分量化
    - 神经网络压缩

主要组件:
    - STEFunction: PyTorch自定义autograd函数
    - ste: 便捷的STE包装函数

参考文献:
    Bengio, Y., Léonard, N., & Courville, A. (2013). 
    Estimating or propagating gradients through stochastic neurons 
    for conditional computation.
"""

import typing as tp

import torch

__all__ = ["ste"]


class STEFunction(torch.autograd.Function):
    """直通估计器的PyTorch自定义函数。
    
    实现了前向传播执行任意函数，反向传播直接传递梯度的机制。
    这是量化感知训练的核心组件。
    
    Example:
        >>> # 使用STE进行四舍五入
        >>> x = torch.tensor([1.4, 2.6, 3.5], requires_grad=True)
        >>> y = STEFunction.apply(x, torch.round)
        >>> print(y)  # tensor([1., 3., 4.])
        >>> y.sum().backward()
        >>> print(x.grad)  # tensor([1., 1., 1.])  # 梯度直接传递
    """

    @staticmethod
    def forward(
        ctx: tp.Any, 
        tensor: torch.Tensor, 
        fn: tp.Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """前向传播：执行指定的函数。
        
        Args:
            ctx: PyTorch上下文对象，用于保存反向传播所需的信息。
            tensor: 输入张量。
            fn: 要执行的函数（如torch.round、torch.floor等）。
            
        Returns:
            函数fn应用于tensor的结果。
        """
        return fn(tensor)

    @staticmethod
    def backward(
        ctx: tp.Any, 
        grad_output: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, None]:
        """反向传播：直接传递梯度。
        
        STE的核心：忽略前向函数的实际导数，直接将输出梯度传递给输入。
        这使得不可微的量化操作可以参与梯度下降优化。
        
        Args:
            ctx: PyTorch上下文对象。
            grad_output: 输出的梯度。
            
        Returns:
            元组(输入梯度, None)。第二个None对应fn参数，它不需要梯度。
        """
        # 直接传递梯度，不做任何修改
        return grad_output, None


def ste(
    tensor: torch.Tensor, 
    fn: tp.Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """直通估计器便捷函数。
    
    对输入张量应用指定函数，同时保持梯度的直接传递。
    这是量化感知训练中处理不可微操作的标准方法。

    Args:
        tensor (`torch.Tensor`):
            输入张量。
        fn (`Callable[[torch.Tensor], torch.Tensor]`):
            要应用的函数。常见选择包括：
            - torch.round: 四舍五入
            - torch.floor: 向下取整
            - torch.ceil: 向上取整
            - 自定义量化函数

    Returns:
        `torch.Tensor`:
            fn(tensor)的结果，但梯度会直接传递。
            
    Example:
        >>> # 量化感知训练中的使用
        >>> x = torch.randn(4, 4, requires_grad=True)
        >>> scale = 0.1
        >>> # 量化：缩放 -> 四舍五入 -> 反缩放
        >>> q = ste(x / scale, torch.round) * scale
        >>> # 反向传播时，梯度会穿过round操作
        >>> loss = (q - x).pow(2).mean()
        >>> loss.backward()
        >>> print(x.grad is not None)  # True
        
    Note:
        - STE是一种近似方法，可能导致训练不稳定
        - 对于大的量化误差，STE的近似可能不准确
        - 在实践中，STE通常与其他技术（如渐进量化）结合使用
    """
    return STEFunction.apply(tensor, fn)  # type: ignore
