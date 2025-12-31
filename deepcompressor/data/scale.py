# -*- coding: utf-8 -*-
"""量化缩放因子模块。

本模块定义了量化过程中使用的缩放因子(scale)数据结构。
缩放因子是量化的核心参数，用于将浮点数映射到量化范围。

量化公式:
    q = round(x / scale)  # 量化
    x' = q * scale        # 反量化

在多级量化（如分组量化）中，缩放因子可以形成层次结构：
    - 全局缩放因子：应用于整个张量
    - 分组缩放因子：应用于张量的子组
    - 叶子缩放因子：最细粒度的缩放因子

主要组件:
    - QuantScale: 层次化的量化缩放因子类
"""

import typing as tp

import torch

__all__ = ["QuantScale"]


class QuantScale:
    """层次化的量化缩放因子类。
    
    支持多级量化场景，如分组量化(group quantization)中的层次缩放因子。
    可以包含子缩放因子(children)或叶子缩放因子(leaves)。
    
    层次结构示例:
        对于 W4A8KV4 量化，权重可能有如下缩放因子结构：
        - 全局缩放因子 (per-tensor)
          └── 分组缩放因子 (per-group, 如每128个元素一组)
              └── 叶子缩放因子 (实际的缩放值)
    
    Attributes:
        data: 合并后的缩放因子张量
        _children: 子QuantScale列表（用于多级量化）
        _leaves: 叶子缩放因子张量列表
    
    Example:
        >>> scale = QuantScale()
        >>> scale.append(torch.tensor([0.1, 0.2]))  # 添加叶子缩放因子
        >>> scale.append(torch.tensor([0.15, 0.25]))
        >>> print(scale.is_quantized())  # True
    """
    
    data: torch.Tensor                  # 合并后的缩放因子数据
    _children: list["QuantScale"]       # 子缩放因子列表
    _leaves: list[torch.Tensor]         # 叶子缩放因子列表

    def __init__(self):
        """初始化空的量化缩放因子。"""
        self.data, self._children, self._leaves = None, [], []  # type: ignore

    @property
    def num_children(self) -> int:
        """获取子缩放因子的数量。"""
        return len(self._children)

    @property
    def num_leaves(self) -> int:
        """获取叶子缩放因子的数量。"""
        return len(self._leaves)

    def is_quantized(self) -> bool:
        """检查缩放因子是否已设置（即是否已完成量化）。
        
        Returns:
            如果缩放因子数据存在且所有子节点都已量化，返回True。
        """
        return self.data is not None and bool(self._leaves or all(child.is_quantized() for child in self._children))

    def get_child(self, index: int) -> "QuantScale":
        """获取指定索引的子缩放因子。
        
        Args:
            index: 子缩放因子的索引。
            
        Returns:
            指定索引的子QuantScale对象。
        """
        return self._children[index]

    def append(self, scale: tp.Union[torch.Tensor, "QuantScale"]) -> "QuantScale":
        """添加一个缩放因子。
        
        可以添加张量（作为叶子）或QuantScale对象（作为子节点）。
        添加时会自动合并到全局缩放因子data中。
        
        Args:
            scale: 要添加的缩放因子，可以是张量或QuantScale对象。
            
        Returns:
            self，支持链式调用。
            
        Raises:
            AssertionError: 如果尝试向非叶子节点添加张量，或向叶子节点添加QuantScale。
            TypeError: 如果scale类型不支持。
        """
        if isinstance(scale, torch.Tensor):
            assert not self._children, "Cannot append a tensor scale to a non-leaf QuantScale."
            # 将新的缩放因子与现有的全局缩放因子合并
            self.data = _join_scale_tensor(self.data, scale)
            self._leaves.append(scale)
        elif isinstance(scale, QuantScale):
            assert not self._leaves, "Cannot append a non-leaf QuantScale to a leaf QuantScale."
            self.data = _join_scale_tensor(self.data, scale.data)
            self._children.append(scale)
        else:
            raise TypeError(f"Unsupported scale type: {type(scale)}")
        return self

    def extend(self, scale: "QuantScale") -> "QuantScale":
        """扩展当前缩放因子，合并另一个QuantScale的内容。
        
        Args:
            scale: 要合并的QuantScale对象。
            
        Returns:
            self，支持链式调用。
        """
        self.data = _join_scale_tensor(self.data, scale.data)
        if scale._children:
            assert not self._leaves, "Cannot extend a leaf QuantScale with a non-leaf QuantScale."
            self._children.extend(scale._children)
        elif scale._leaves:
            assert not scale._children, "Cannot extend a non-leaf QuantScale with a leaf QuantScale."
            self._leaves.extend(scale._leaves)
        return self

    def join(self, scale: "QuantScale") -> "QuantScale":
        """创建一个新的QuantScale，包含当前对象和另一个对象作为子节点。
        
        Args:
            scale: 要连接的QuantScale对象。
            
        Returns:
            新的QuantScale对象，包含两个子节点。
        """
        return QuantScale().append(self).append(scale)

    def remove_zero(self) -> "QuantScale":
        """移除零值缩放因子（将其替换为1）。
        
        零值缩放因子会导致除零错误，此方法将其替换为1以避免问题。
        
        Returns:
            self，支持链式调用。
        """
        self.data[self.data == 0] = 1
        return self

    def state_dict(
        self,
        param_name: str,
        device: torch.device | str = "cpu",
        flatten: bool = True,
        level_base: int = 0,
    ) -> dict[str, torch.Tensor]:
        """获取状态字典，用于模型保存和加载。
        
        Args:
            param_name: 参数名称前缀。
            device: 目标设备。
            flatten: 是否展平层次结构。
            level_base: 层级基数（用于生成唯一键名）。
            
        Returns:
            包含所有缩放因子的字典，键为参数名，值为张量。
        """
        if self._children:
            state_dict = {}
            for i, child in enumerate(self._children):
                child_param_name = param_name if flatten else f"{param_name}.{i}"
                child_level_base = len(state_dict) if flatten else 0
                child_state_dict = child.state_dict(child_param_name, device, flatten, child_level_base)
                state_dict.update(child_state_dict)
            return state_dict
        else:
            return {f"{param_name}.{level_base + i}": leaf.to(device) for i, leaf in enumerate(self._leaves)}


def _join_scale_tensor(global_scale: torch.Tensor | None, local_scale: torch.Tensor) -> torch.Tensor:
    """将局部缩放因子与全局缩放因子相乘合并。
    
    在多级量化中，最终的缩放因子是各级缩放因子的乘积。
    此函数处理不同形状的缩放因子的广播和相乘。

    Args:
        global_scale (`torch.Tensor` or `None`):
            全局缩放因子张量。形状为 (#gs_g0, 1, #gs_g1, 1, #gs_g2, 1, ...)
            其中 #gs_gi 表示第i维的分组数。
        local_scale (`torch.Tensor`):
            局部缩放因子张量。形状为 (#ss_g0, 1, #ss_g1, 1, #ss_g2, 1, ...)

    Returns:
        `torch.Tensor`:
            合并后的缩放因子张量，保持local_scale的原始形状。
            
    Note:
        合并过程：
        1. 将local_scale重塑为与global_scale兼容的形状
        2. 执行逐元素乘法
        3. 恢复原始形状
    """
    # global_scale: (#gs_g0, 1, #gs_g1, 1, #gs_g2, 1, ...)
    # local_scale:  (#ss_g0, 1, #ss_g1, 1, #ss_g2, 1, ...) -> (#gs_g0, rs0, #gs_g1, rs1, #gs_g2, rs2, ...)
    shape = local_scale.shape
    return (
        local_scale
        if global_scale is None
        else local_scale.view(
            tuple(
                global_scale.shape[i] if j == 0 else local_scale.shape[i] // global_scale.shape[i]
                for i in range(0, len(global_scale.shape), 2)
                for j in range(2)
            )
        ).mul(global_scale)
    ).view(shape)
