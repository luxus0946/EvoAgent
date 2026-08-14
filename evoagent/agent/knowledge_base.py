"""轻量 RAG 知识库：算法使用指南与最佳实践。

设计文档指定 Chroma + BGE 嵌入向量库，此处提供轻量实现：
- 关键词 TF 检索（无重型依赖）
- 接口与向量库对齐（add/retrieve），后续可平滑替换为 Chroma
"""

from dataclasses import dataclass, field

# 内置知识文档：各优化算法的使用指南
_KNOWLEDGE_DOCS: list[str] = [
    "CMA-ES 适合连续空间黑盒优化的局部精调：步长 sigma 初始取参数范围的 10%-25%，"
    "目标函数平滑或近凸时收敛极快，多峰时建议配合全局探索阶段使用。",
    "贝叶斯优化在小样本（<200 次评估）下效率最高，但高维（>10）时代理模型误差增大，"
    "适合作为精调阶段而非全局搜索阶段。",
    "遗传算法在 10 维以上多峰问题中比 CMA-ES 更稳健，变异率 0.1-0.2 平衡探索与收敛，"
    "适合没有梯度信息且评估廉价的场景。",
    "模拟退火通过概率接受劣解逃离局部最优，初始温度应接近目标值尺度，"
    "冷却系数 0.99 附近兼顾速度与质量。",
    "随机搜索是无偏基线：在评估预算充足时应作为探索补充，但单独使用时效率低。",
    "多峰含噪问题的最优做法：先用探索类工具（随机/模拟退火）粗搜定位有希望区域，"
    "再用 CMA-ES 或贝叶斯优化精调；早停阈值设置在 20%-40% 预算之间较为稳健。",
]


@dataclass
class KnowledgeBase:
    """轻量关键词检索知识库。"""

    docs: list[str] = field(default_factory=lambda: list(_KNOWLEDGE_DOCS))

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """按关键词 TF 分数检索最相关文档。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            相关文档列表
        """
        if not query.strip():
            return []
        terms = _tokenize(query)
        scored = []
        for doc in self.docs:
            doc_terms = _tokenize(doc)
            score = sum(doc_terms.count(t) for t in terms)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda kv: -kv[0])
        return [doc for _, doc in scored[:top_k]]


def _tokenize(text: str) -> list[str]:
    """极简分词：拉丁词 + 双字中文 n-gram。"""
    import re

    tokens: list[str] = []
    tokens += re.findall(r"[a-zA-Z0-9_\-\.]+", text.lower())
    cn = re.findall(r"[\u4e00-\u9fff]+", text)
    for seg in cn:
        tokens += [seg[i : i + 2] for i in range(len(seg) - 1)]
    return tokens
