"""
reranker — BGE-Reranker 精排 + 完整检索 Pipeline

本模块负责知识库检索链路的"精排"环节，与 knowledge_base.py 形成互补：

  knowledge_base.py  — 嵌入（BGE-M3）+ 粗排（Hybrid 混合检索）
  reranker.py        — 精排（BGE-Reranker CrossEncoder）+ 检索一体化入口

完整检索链路（retrieve 函数）：
  用户提问
    → BGE-M3 编码（dense + sparse）
    → Milvus Hybrid 召回 Top-K 候选（粗排：WeightedRanker 融合）
    → BGE-Reranker CrossEncoder 精排（候选逐个打分）
    → 返回 Top-N 结果 + 置信度

置信度门控策略：
  confidence >= 0.75 → 高置信度：知识库内容可直接回答，走 LLM 生成
  confidence < 0.75  → 低置信度：知识库覆盖不足，触发 Web 搜索兜底

"""

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              标准库导入                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

import os                              # 路径拼接、文件存在性检查
from dataclasses import dataclass      # @dataclass — 精简数据结构定义
from typing import Optional            # 可选类型注解

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              第三方库导入                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

import torch                           # PyTorch — 检测 CUDA 可用性、设备选择
from sentence_transformers import CrossEncoder  # HuggingFace 的 CrossEncoder 封装

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              项目内部模块                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

from backend.config import get_settings    # 全局配置 — 获取 Reranker 模型路径
from backend.core.logger import get_logger # 结构化日志

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              模块级变量                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

logger = get_logger(__name__)
# __name__ = "backend.core.reranker"

backend_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# 用于拼接本地模型绝对路径

RERANK_MAX_INPUT_CHARS = 1200
# 精排时单篇文档的最大字符数（截断保护）
# CrossEncoder max_length=512 tokens，中文约 1.5~2 字符/token
# 1200 字符 ≈ 600~800 tokens，给 (query + document) 拼接后留足空间
# 同时避免过长文档浪费算力——精排只需判断相关性，不需要全文


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                 RankedDocument — 精排后的单个文档结果                          ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class RankedDocument:
    """
    精排后返回的单个文档，字段用于 LLM 生成阶段。

    与 knowledge_base.py 中 DocumentChunk 的区别：
      DocumentChunk   — 写入 Milvus 的原始数据（含向量）
      RankedDocument  — 检索链路末端的精简结果（含精排分数）
    """
    content:        str    # 文档文本（截断后），LLM 直接引用
    score:          float  # BGE-Reranker 相关性分数 [0, 1]
                           # CrossEncoder sigmoid 输出，越接近 1 越相关
    original_index: int    # 在原始 Hybrid 召回列表中的位置（0 起）
                           # 保留此字段用于调参对比：召回排第 N，精排后变成第几
    metadata:       dict   # 来源元数据，含 source_name / chunk_type / course_id / document_id / chunk_index


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║              BGE-Reranker 精排服务（进程内单例）                               ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class BGEReranker:
    """
    BGE-Reranker 精排服务 — CrossEncoder 架构，进程内单例。

    为什么需要 Reranker（精排）？
      Hybrid 混合检索的 score 是"向量距离的加权和"，不是真正的"语义相关性"。
      向量检索适合快速粗筛（从海量库中找出"可能相关"的候选），
      但排序精度有限——COSINE 0.85 和 COSINE 0.82 的两篇文章，哪个更相关？
      向量相似度无法精确区分，需要 Reranker 真正"阅读"两段文字再打分。

    CrossEncoder vs Bi-Encoder：
      Bi-Encoder（如 BGE-M3）：
        Query 和 Document 独立编码成向量，再算相似度。
        优点：快（Document 向量可以预计算存库），适合海量粗筛。
        缺点：Query 和 Document 没交互，精度有限。
      CrossEncoder（如 BGE-Reranker）：
        将 (Query, Document) 拼接后一起输入模型，同时做注意力计算。
        优点：Query 和 Document 有深度交互，精度高。
        缺点：慢（每个 (query, doc) 对都要过模型），只能对少量候选精排。
      → 两者的关系：Bi-Encoder 粗筛 → CrossEncoder 精排，互补。

    模型选择：
      优先使用本地模型（配置中的 reranker_model_path），
      本地不存在时自动从 HuggingFace 下载 "BAAI/bge-reranker-v2-m3"。
      v2-m3 是多语言版本，支持中英文混合检索。

    sigmoid 激活：
      CrossEncoder 默认 sigmoid 输出层，predict() 直接返回 [0, 1] 概率。
      这个概率可解释为"该文档与 Query 相关的置信度"，无需额外归一化。

    用法：
        reranker = BGEReranker.get_instance()
        docs, confidence = reranker.rerank_with_confidence(
            query="什么是 Spring IOC？",
            documents=candidates,  # Hybrid 召回的候选
            top_k=3,
        )
    """

    # ── 单例持有 ─────────────────────────────────────────────────
    _instance: Optional["BGEReranker"] = None
    # 进程内唯一实例引用，None 表示尚未加载模型

    def __init__(self):
        """
        加载 BGE-Reranker 模型。

        模型加载策略（本地优先）：
          1. 先检查配置中的本地模型路径是否存在且包含模型文件
          2. 存在 → 从本地加载（离线环境可用）
          3. 不存在 → 从 HuggingFace 自动下载 "BAAI/bge-reranker-v2-m3"

        ACCELERATE_USE_META_DEVICE = "0"：
          禁止 accelerate 的 meta device 模式。
          这是一项兼容性设置，防止在某些 accelerate 版本下模型参数
          被加载到 meta device 导致无法正常推理。

        设备选择：
          CUDA 可用 → GPU 推理（快，约 20~50ms/对）
          CUDA 不可用 → CPU 推理（慢，约 100~300ms/对，但仍可接受）
        """
        # ── 禁止 meta device 模式（兼容性）───────────────────────
        os.environ["ACCELERATE_USE_META_DEVICE"] = "0"

        settings = get_settings()
        model_path = os.path.normpath(os.path.join(backend_path, settings.reranker_model_path))

        # ── 检查本地模型是否存在 ─────────────────────────────────
        use_local = (
            os.path.exists(model_path)
            and os.path.isdir(model_path)
            and any(
                # 模型文件后缀：.bin（PyTorch 格式）、.safetensors（安全格式）、
                # .json（配置文件、tokenizer）
                f.endswith((".bin", ".safetensors", ".json"))
                for f in os.listdir(model_path)
            )
        )

        # ── 确定最终加载路径 ─────────────────────────────────────
        model_id = model_path if use_local else "BAAI/bge-reranker-v2-m3"
        # 本地模型 → 绝对路径；HuggingFace → 模型 ID，自动下载到缓存

        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("reranker.loading", model_id=model_id, device=device)
        self._model = CrossEncoder(
            model_id,
            device=device,
            max_length=512,  # BGE-Reranker 最大 token 数，覆盖绝大多数 query+doc 对
        )
        logger.info("reranker.loaded", model_id=model_id)

    # ── 单例入口 ─────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "BGEReranker":
        """
        获取精排模型单例。

        首次调用：加载模型到内存/显存（约 2~5 秒，模型约 2.2GB）
        后续调用：直接返回缓存实例（毫秒级）
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── 精排方法 ─────────────────────────────────────────────────

    def rerank_with_confidence(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3,
    ) -> tuple[list[RankedDocument], float]:
        """
        对候选文档做 CrossEncoder 精排，返回 Top-K 结果 + 置信度。

        处理流程：
          1. 构造 (query, document) 输入对
          2. 截断过长文档（> RERANK_MAX_INPUT_CHARS 的截断）
          3. CrossEncoder.predict() 批量推理 → [0, 1] 相关性分数
          4. 按分数降序排列
          5. 返回 Top-K + Top-1 置信度

        置信度的含义：
          Top-1 文档的 BGE-Reranker 相关性分数。
          这是 sigmoid 概率，可解释为"模型认为该文档与 Query 相关的概率"。

        置信度门控策略（由调用方实现）：
          ≥ 0.75  → 知识库内容质量好，直接走 LLM 生成
          < 0.75  → 知识库覆盖不足，触发 Web 搜索兜底
          0.75 是一个经验阈值，可根据业务场景调整：
            - 医疗/法律等高风险场景 → 提高到 0.85
            - 闲聊/娱乐等低风险场景 → 降低到 0.6

        Args:
            query:      用户原始提问
            documents:  Hybrid 召回候选列表，每项需含 "content" 和 "metadata"
            top_k:      精排后返回的文档数量（默认 3，LLM 上下文窗口有限）

        Returns:
            (ranked_docs, confidence) 二元组：
              ranked_docs — 按相关性降序排列的 RankedDocument 列表（长度 ≤ top_k）
              confidence  — Top-1 文档的 BGE 置信度 [0, 1]
        """
        if not documents:
            return [], 0.0

        # ── 构造 CrossEncoder 输入对 ─────────────────────────────
        # 每对格式：(query_text, document_text)
        # 截断过长文档到 RERANK_MAX_INPUT_CHARS，防止超出 max_length=512 tokens
        pairs = [
            (query, (doc.get("content") or "")[:RERANK_MAX_INPUT_CHARS])
            for doc in documents
        ]

        # ── 批量推理 ─────────────────────────────────────────────
        # CrossEncoder.predict() 内部：
        #   1. Tokenize：(query, doc) → input_ids + attention_mask + token_type_ids
        #   2. 前向传播 → logits
        #   3. sigmoid(logits) → [0, 1] 概率
        # 返回 numpy array，tolist() 转为 Python list[float]
        scores: list[float] = self._model.predict(pairs).tolist()

        # ── 构造 RankedDocument 并按分数排序 ─────────────────────
        ranked = sorted(
            [
                RankedDocument(
                    content=documents[i].get("content", ""),
                    score=scores[i],
                    original_index=i,                         # 保留召回位置
                    metadata=documents[i].get("metadata", {}),
                )
                for i in range(len(documents))
            ],
            key=lambda x: x.score,
            reverse=True,  # 降序：分数最高的排最前面
        )

        top_results = ranked[:top_k]
        confidence = top_results[0].score if top_results else 0.0

        logger.info(
            "reranker.done",
            candidates=len(documents),
            top_k=top_k,
            confidence=round(confidence, 4),
        )

        return top_results, confidence


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                  retrieve() — 检索全链路一体化入口                            ║
# ║                                                                            ║
# ║  对外唯一调用入口，封装三个步骤，调用方无需感知向量和 Milvus 细节：            ║
# ║    ① BGE-M3 编码（内部调用 BGEMEmbedder）                                   ║
# ║    ② Hybrid 混合召回（内部调用 KnowledgeBaseClient._hybrid_search）           ║
# ║    ③ BGE-Reranker 精排（内部调用 BGEReranker.rerank_with_confidence）        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def retrieve(
    query: str,
    tenant_id: str,
    course_id: Optional[str] = None,
    recall_top_k: int = 10,
    rerank_top_k: int = 3,
) -> tuple[list[RankedDocument], float]:
    """
    检索全链路：Hybrid 粗排 → BGE 精排，一步完成。

    设计意图：
      调用方（API 层 / Agent）只需传入 Query 文本和租户参数，
      向量编码、Milvus 检索、精排全部在内部完成，降低调用方的认知负担。

    为什么用延迟导入（lazy import）？
      BGEMEmbedder 和 KnowledgeBaseClient 都在 knowledge_base.py 中，
      如果在本模块顶部 import，可能与 knowledge_base 形成循环依赖。
      在函数内部 import 只在实际调用时才加载，安全且不影响启动速度。

    Args:
        query:         用户原始提问
        tenant_id:     租户 ID（必填，从 JWT 中提取，保证多租户隔离）
        course_id:     课程 ID（可选，传了则只检索该课程的内容）
        recall_top_k:  Hybrid 粗排召回数量（默认 10，给精排的候选池大小）
        rerank_top_k:  精排后返回数量（默认 3，最终传给 LLM 的文档数）

    Returns:
        (ranked_docs, confidence) 二元组：
          ranked_docs — 精排后 Top-rerank_top_k 文档，按相关性降序
          confidence  — Top-1 文档的 BGE 置信度 [0, 1]

    Example:
        docs, conf = retrieve(
            query="Java 中 IOC 容器的作用？",
            tenant_id="tenant_default",
            course_id="3e76aeed-5e01-4aa7-be8d-2055d12b9ea7",
        )
        if conf >= 0.75:
            # 高置信度，用知识库内容 + LLM 生成回答
            context = "\n".join(d.content for d in docs)
            prompt = f"根据以下资料回答问题：\n{context}\n\n问题：{query}"
        else:
            # 低置信度，触发 Web 搜索兜底
            web_results = web_search(query)
    """
    # ── 延迟导入（避免循环依赖）─────────────────────────────────
    from backend.core.knowledge_base import BGEMEmbedder, KnowledgeBaseClient

    # ── 第一步：BGE-M3 编码 ─────────────────────────────────────
    # 将用户自然语言问题转为 dense + sparse 双向量
    embedder = BGEMEmbedder.get_instance()
    dense_vec, sparse_vec = embedder.encode_query(query)

    # ── 第二步：Hybrid 混合召回 ─────────────────────────────────
    # Dense（语义）+ Sparse（关键词）→ WeightedRanker 融合 → 粗排候选
    kb = KnowledgeBaseClient()
    filters = kb._build_filter(tenant_id, course_id)
    candidates = kb._hybrid_search(
        query_embedding=dense_vec,
        query_sparse=sparse_vec,
        top_k=recall_top_k,
        filters=filters,
    )

    if not candidates:
        logger.info("retrieve.empty", query_preview=query[:50])
        return [], 0.0

    # ── 第三步：BGE-Reranker 精排 ───────────────────────────────
    # CrossEncoder 逐个 (query, doc) 打分 → 排序 → 返回 Top-K + 置信度
    reranker = BGEReranker.get_instance()
    return reranker.rerank_with_confidence(query, candidates, top_k=rerank_top_k)


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              快速自测                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    """
    快速验证精排链路。

    测试一：BGEReranker 纯精排（静态候选，不依赖 Milvus）
      目的：验证 Reranker 模型加载和打分逻辑是否正常。
      用几个手动构造的候选文档，看排序是否合理。

    测试二：retrieve() 全链路（需要 Milvus 已建库，有实际数据）
      目的：验证从编码到召回再到精排的完整流程。

    运行方式：
      python -m backend.core.reranker
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from dotenv import load_dotenv
    load_dotenv(".env.local")

    # ═══════════════════════════════════════════════════════════════
    # 测试一：精排（静态候选，不依赖 Milvus）
    # ═══════════════════════════════════════════════════════════════

    query = "商品聚合大模型中 Hard Negative Sampling 怎么实现？"
    candidates = [
        {
            "content": "Hard Negative Sampling 是一种负样本挖掘策略，选取与正样本相似但标签不同的样本作为难负例，强迫模型学习细粒度区分能力。在商品召回场景中，通常从 Top-K 召回结果中排除正样本后取前几条作为难负例。",
            "metadata": {"source_name": "sample2 > Hard Negative Sampling"},
        },
        {
            "content": "双塔召回模型（Bi-Encoder）将用户和商品分别编码为向量，通过内积计算相关性。模型训练阶段使用 In-Batch Negative 策略，将同批次的其他商品作为随机负样本。",
            "metadata": {"source_name": "sample2 > 双塔召回"},
        },
        {
            "content": "Qwen VL 是阿里巴巴开源的视觉语言大模型，支持图文混合输入。微调时需要准备图文对数据，使用 LlamaFactory 或 MS-SWIFT 框架进行 LoRA 微调。",
            "metadata": {"source_name": "sample2 > Qwen VL 微调"},
        },
        {
            "content": "PEFT（Parameter-Efficient Fine-Tuning）是一类只更新少量参数的微调方法，LoRA 是其中最常用的变体，通过低秩矩阵分解减少可训练参数数量。",
            "metadata": {"source_name": "sample2 > PEFT/LoRA"},
        },
    ]

    reranker = BGEReranker.get_instance()
    ranked_docs, confidence = reranker.rerank_with_confidence(
        query=query,
        documents=candidates,
        top_k=3,
    )

    print(f"置信度：{confidence:.4f}（{'高置信度' if confidence >= 0.75 else '低置信度'}）\n")
    for i, doc in enumerate(ranked_docs):
        print(f"[{i+1}] score={doc.score:.4f}  来源：{doc.metadata.get('source_name', '')}")
        print(f"     {doc.content[:80]}\n")

    # ═══════════════════════════════════════════════════════════════
    # 测试二：retrieve() 全链路（需 Milvus 已有知识库数据）
    # ═══════════════════════════════════════════════════════════════

    print("\n--- retrieve() Pipeline 测试 ---")
    ranked_docs2, confidence2 = retrieve(
        query="LlamaFactory 怎么做 Qwen VL 微调？",
        tenant_id="tenant_default",
        course_id=None,   # None = 不限课程，检索所有课程的内容
    )
    print(f"置信度：{confidence2:.4f}（{'高置信度' if confidence2 >= 0.75 else '低置信度'}）\n")
    for i, doc in enumerate(ranked_docs2):
        print(f"[{i+1}] score={doc.score:.4f}  来源：{doc.metadata.get('source_name', '')}")
        print(f"     {doc.content[:80]}\n")
