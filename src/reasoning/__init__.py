"""
超图推理模块 (Hyper_Reasoning)

将图匹配从"硬规则 GED"转换为"软投票 BP"：
1. 目标统一解析：Object/Image/Text → 结构化目标模板
2. 动态超边生成：LLM 即时推理功能区
3. 因子图构建：一元 + 二元 + 高阶因子
4. 信念传播：迭代消息传递，输出节点后验概率
"""

from .hypergraph_bp import HyperGraphReasoner, HyperEdge, FactorGraph, BeliefPropagation

__all__ = ['HyperGraphReasoner', 'HyperEdge', 'FactorGraph', 'BeliefPropagation']
