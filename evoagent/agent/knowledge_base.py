"""Lightweight RAG knowledge base: algorithm usage guides and best practices.

The design document specifies a Chroma + BGE embedding vector store; this module
provides a lightweight implementation:
- Keyword TF retrieval (no heavy dependencies)
- Interface aligned with a vector store (add/retrieve), allowing a smooth swap to Chroma later
"""

from dataclasses import dataclass, field

# Built-in knowledge documents: usage guides for each optimization algorithm
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
    """Lightweight keyword-retrieval knowledge base."""

    docs: list[str] = field(default_factory=lambda: list(_KNOWLEDGE_DOCS))

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve the most relevant documents by keyword TF score.

        Args:
            query: Query text
            top_k: Number of documents to return

        Returns:
            List of relevant documents
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
    """Minimal tokenizer: Latin words plus two-character Chinese n-grams."""
    import re

    tokens: list[str] = []
    tokens += re.findall(r"[a-zA-Z0-9_\-\.]+", text.lower())
    cn = re.findall(r"[\u4e00-\u9fff]+", text)
    for seg in cn:
        tokens += [seg[i : i + 2] for i in range(len(seg) - 1)]
    return tokens
