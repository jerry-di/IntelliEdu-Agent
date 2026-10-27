"""
knowledge_base — BGE-M3 向量嵌入 + Milvus 混合检索

整个知识库系统的核心模块，覆盖三个环节：
  1. 嵌入（Embedding）  — BGE-M3 本地模型，dense + sparse 双向量输出
  2. 存储（Storage）    — Milvus 向量数据库，单 Collection 多租户
  3. 检索（Retrieval）  — 混合检索（Dense 语义 + Sparse 关键词），WeightedRanker 融合

数据流：
  原始文档 → 分块 → BGE-M3 编码 → DocumentChunk → Milvus upsert
  用户提问 → BGE-M3 编码 → _hybrid_search → 候选 chunk 列表

"""

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              标准库导入                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

import hashlib                         # MD5 哈希 — 为 chunk 生成确定性唯一 ID
import os                              # 路径拼接、环境变量
import time                            # Unix 时间戳 — 记录 chunk 写入时间
from dataclasses import dataclass, field  # @dataclass — 简化数据结构定义
from typing import Optional            # 可选类型注解

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              第三方库导入                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

from pymilvus import (                  # Milvus Python SDK
    MilvusClient,                       # 高层客户端（MilvusClient 版本，非 ORM 版本）
    AnnSearchRequest,                   # 单路 ANN 检索请求（dense 或 sparse 各一个）
    WeightedRanker,                     # 多路检索结果加权融合排序器
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              项目内部模块                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

from backend.config import get_settings    # 全局配置单例（模型路径、Milvus 连接参数等）
from backend.core.logger import get_logger # 结构化日志（支持 JSON 格式、上下文绑定）

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              模块级变量                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

logger = get_logger(__name__)
# __name__ = "backend.core.knowledge_base"，日志中显示来源模块

backend_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# __file__ 是 knowledge_base.py 的绝对路径
# 一次 dirname → core/，两次 dirname → backend/，三次 dirname → 项目根目录
# 用途：拼接 BGE-M3 本地模型路径（项目根目录 + 配置中的相对路径）

COLLECTION_NAME = "knowledge_domain"
# Milvus 集合名称，与 init_milvus.py 保持一致
# 单集合设计：所有课程、租户的 chunk 都存在同一集合中
# 通过 tenant_id / course_id 标量字段过滤实现多租户隔离和按课程检索


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║        BGE-M3 本地嵌入模型（进程内单例，dense + sparse 双向量输出）            ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class BGEMEmbedder:
    """
    BGE-M3 本地嵌入模型 — 进程内单例。

    BGE-M3 是 BAAI 开源的多语言嵌入模型，核心特点是一次推理同时输出三种向量：
      - Dense 向量（1024 维 float）  → 语义相似度检索（"意思差不多"的也能召回）
      - Sparse 向量（词权重词典）     → 关键词精确匹配（术语、专有名词不丢）
      - ColBERT 多向量（本项目不用）  → token 级细粒度匹配，内存开销大

    进程内单例模式（Singleton）：
      - 首次调用 get_instance() 加载模型到内存/显存（约 5-15 秒）
      - 后续调用直接返回同一实例（毫秒级），不重复加载
      - 避免多实例导致显存/内存爆炸（BGE-M3 约 2.2GB）

    用法：
        embedder = BGEMEmbedder.get_instance()
        dense, sparse = embedder.encode_query("什么是 Spring IOC？")

    硬件适配：
      - CUDA（NVIDIA 显卡）：fp16 半精度推理，速度快
      - CPU / MPS（Apple M 芯片）：fp32 全精度推理，稳定不崩溃
    """

    # ── 单例持有 ─────────────────────────────────────────────────
    _instance: Optional["BGEMEmbedder"] = None
    # 类级别属性，所有实例共享
    # None = 尚未加载模型，非 None = 已加载的唯一实例引用

    def __init__(self, model_path: str):
        """
        构造函数 — 私有使用，外部禁止直接调用，通过 get_instance() 间接调用。

        职责：
          1. 打兼容性补丁（适配不同 transformers / FlagEmbedding 版本）
          2. 加载 BGE-M3 模型到内存/显存
        """

        # ── 兼容性补丁 ①：FlagEmbedding 依赖 transformers.is_torch_fx_available ──
        # transformers >= 5.0 移除了该函数，但当前锁定 transformers==4.51.0 不受影响。
        # 此补丁作为保险，避免未来升级时报 ImportError 导致整个服务启动失败。
        import importlib.util as _ilu
        from transformers.utils import import_utils as _tf_iu
        if not hasattr(_tf_iu, "is_torch_fx_available"):
            _tf_iu.is_torch_fx_available = (
                lambda: _ilu.find_spec("torch.fx") is not None
            )

        # ── 兼容性补丁 ②：XLMRobertaModel 不接受 dtype 关键字参数 ──
        # FlagEmbedding 内部调用 XLMRobertaModel(config, dtype=...) ，
        # 但某些 transformers 版本的 __init__ 签名中没有 dtype 形参，会抛 TypeError。
        # 通过猴子补丁子类在传给父类前丢弃 dtype 参数。
        from transformers.models.xlm_roberta import modeling_xlm_roberta as _xlm
        _OriginalXLMRoberta = _xlm.XLMRobertaModel

        class _PatchedXLMRobertaModel(_OriginalXLMRoberta):
            """子类覆盖 __init__：在调用父类前丢弃 FlagEmbedding 传入的 dtype 参数"""
            def __init__(self, config, **kwargs):
                kwargs.pop("dtype", None)          # 丢弃 dtype，父类不认识它
                super().__init__(config, **kwargs) # 正常初始化

        _xlm.XLMRobertaModel = _PatchedXLMRobertaModel  # 全局替换

        # ── 加载 BGE-M3 模型 ─────────────────────────────────────
        import torch
        from FlagEmbedding import BGEM3FlagModel

        logger.info("bge_m3.loading", model_path=model_path)

        # fp16 仅在 CUDA 上安全启用
        # MPS（Apple M 系列芯片）在 BGE-M3 attention 矩阵乘法上会触发 LLVM ERROR
        # CPU 模式下用 fp32，速度稍慢但稳定
        _use_fp16 = torch.cuda.is_available()

        self._model = BGEM3FlagModel(
            model_name_or_path=model_path,
            use_fp16=_use_fp16,
        )
        logger.info("bge_m3.loaded", use_fp16=_use_fp16)

    # ── 公开方法 ─────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "BGEMEmbedder":
        """
        获取模型单例 — 整个进程的唯一入口。

        首次调用：创建实例（加载模型 5~15 秒，取决于磁盘速度）
        后续调用：直接返回缓存实例（毫秒级）

        Returns:
            全局唯一的 BGEMEmbedder 实例
        """
        if cls._instance is None:
            # 拼接模型绝对路径
            # 配置中的 bge_m3_model_path 有两种形式：
            #   相对路径：./models/embedding/bge-m3 → 拼接 backend_path 得到绝对路径
            #   绝对路径：D:/黑马模型分享/models/embedding/bge-m3 → os.path.join 保留绝对路径
            bge3_path = os.path.normpath(os.path.join(backend_path, get_settings().bge_m3_model_path))
            cls._instance = BGEMEmbedder(bge3_path)
        return cls._instance

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
    ) -> tuple[list[list[float]], list[dict]]:
        """
        批量编码文本 → (dense_vecs, sparse_vecs)。

        适用于建库场景：一次传入 N 个 chunk，一批推理，GPU 利用率高。

        为什么需要两种向量？
          ┌────────────┬──────────────────┬─────────────────────┐
          │ 向量类型    │ 维度 / 格式       │ 适合场景             │
          ├────────────┼──────────────────┼─────────────────────┤
          │ Dense      │ 1024 维 float[]  │ 语义相似（"意思相近"）│
          │ Sparse     │ {token_id: wt}   │ 关键词精确匹配       │
          └────────────┴──────────────────┴─────────────────────┘
          混合检索 = 两路并行召回 → WeightedRanker 加权融合 → 取 Top-K

        Args:
            texts:      待编码文本列表，如 ["段落1", "段落2", ...]
            batch_size: 单次推理批大小。越大 GPU 利用率越高但显存压力越大。
                        12 是 16GB 显存 / 统一内存下的经验值。

        Returns:
            (dense_vecs, sparse_vecs) 二元组：
              dense_vecs  — list[list[float]]，每个子列表 1024 个 float
              sparse_vecs — list[dict]，每个字典是 {token_id(int): weight(float)}
        """
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,            # BGE-M3 最长支持 8192 token
            return_dense=True,          # 输出稠密语义向量
            return_sparse=True,         # 输出稀疏词权重向量
            return_colbert_vecs=False,  # ColBERT 多向量 — 本项目不用，关闭省内存
        )

        # numpy ndarray → Python 原生 list
        dense_vecs = output["dense_vecs"].tolist()

        # numpy.float16 → Python 原生 float（这一步非常重要！）
        # 原因：LangGraph MemorySaver 用 msgpack 序列化 State，
        #       msgpack 不认识 numpy.float16，会抛 TypeError。
        # 必须在存入 State 之前全部转为 Python 原生类型。
        sparse_vecs = [
            {int(k): float(v) for k, v in d.items()}
            for d in output["lexical_weights"]
        ]

        return dense_vecs, sparse_vecs

    def encode_query(self, text: str) -> tuple[list[float], dict]:
        """
        编码单条查询文本 → (dense_vec, sparse_vec)。

        与 encode() 的区别：
          encode()       用于建库 — 批量编码 N 个 chunk，batch_size 可调
          encode_query() 用于检索 — 只编码 1 条用户问题，batch_size=1

        单条编码避免不必要的 padding，减少检索延迟（通常 < 100ms）。

        Args:
            text: 用户原始提问，如 "Java 中 IOC 容器的作用是什么？"

        Returns:
            (dense_vec, sparse_vec) 二元组：
              dense_vec  — 1024 维 float 列表（单一向量，非嵌套列表）
              sparse_vec — {token_id(int): weight(float)} 字典
        """
        dense_list, sparse_list = self.encode([text], batch_size=1)
        return dense_list[0], sparse_list[0]


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║           数据类：DocumentChunk（建库写入 Milvus 的最小数据单元）               ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class DocumentChunk:
    """
    准备写入 Milvus 的单个文档块，字段与 knowledge_domain Collection Schema 一一对应。

    一个 "chunk" 的完整生命周期：
      原始文档（PDF/Markdown）
        → 分割成若干小段落（chunk）
        → BGE-M3 编码（dense + sparse 双向量）
        → 组装为 DocumentChunk
        → 写入 Milvus
        → 检索时返回给用户

    字段说明（与 Milvus 字段名保持一致）：

      id               — 主键，MD5(content + document_id + chunk_index)
                         确定性生成 → 同内容同位置永远同 ID → 支持幂等 upsert

      content          — chunk 的文本内容（建库后会被 Contextual RAG 增强）

      embedding        — 稠密向量（BGE-M3 dense，1024 维 float 数组）
                         用于语义相似度检索（COSINE 度量）

      sparse_embedding — 稀疏向量（BGE-M3 lexical weights）
                         {token_id: weight}，用于关键词精确检索（IP 度量）

      course_id        — 所属课程 UUID
                         检索时按此字段过滤，保证不会跨课程串数据

      document_id      — 所属文档 UUID
                         文档更新时：按此字段删旧 → 插新，实现幂等更新

      source_name      — 来源标注，检索结果展示用
                         如 "Java讲义 > 第3章 > 3.1 IOC 容器"

      chunk_type       — 内容类型标记："text"（默认）/ "code" / "table"
                         后续可用于分类型检索（如只搜代码块）

      chunk_index      — 在原文档中的顺序位置（从 0 开始）
                         检索后可用于展示上下文（前一个/后一个 chunk）

      version          — 课程版本号
                         版本升级时配合 course_id 检索最新内容

      tenant_id        — 租户 ID（SaaS 多租户隔离）
                         所有检索必须带 tenant_id 过滤条件

      updated_at       — Unix 时间戳（秒），自动填当前时间
                         用于排查问题、按时间排序等
    """
    id:               str                          # 主键，32 位 MD5 十六进制字符串
    content:          str                          # chunk 文本全文
    embedding:        list[float]                  # 稠密向量，1024 个 float
    sparse_embedding: dict                         # 稀疏向量，{token_id(int): weight(float)}
    course_id:        str                          # 课程 UUID
    document_id:      str                          # 文档 UUID
    source_name:      str                          # 来源路径（检索结果展示用）
    chunk_index:      int                          # 在文档中的顺序位置（0-based）
    version:          str                          # 课程版本号
    chunk_type:       str = "text"                 # 类型标记，默认 "text"
    tenant_id:        str = "tenant_default"       # 租户 ID，默认值
    updated_at:       int = field(default_factory=lambda: int(time.time()))
    # default_factory 保证每次创建对象时重新计算时间戳，而非模块加载时固定


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                           工具函数：生成 Chunk ID                             ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def generate_chunk_id(content: str, document_id: str, chunk_index: int) -> str:
    """
    生成 chunk 的全局唯一 ID（MD5 散列）。

    为什么不用 UUID？
      UUID 是随机的 → 每次建库同一 chunk 的 ID 都不同
      MD5 是确定性的 → 同一内容 + 同一文档 + 同一位置 → 永远生成同一个 ID

    确定性的好处：
      重复建库时，相同内容的 chunk 生成相同主键
      → Milvus upsert 自动覆盖旧数据（而非插入重复数据）
      → 不需要手动删除后再插入，天然幂等

    ID 的组成（拼接后取 MD5）：
      document_id + chunk_index + content[:50]
      - document_id    — 保证不同文档不冲突
      - chunk_index    — 保证同一文档内不同位置不冲突
      - content[:50]   — 取前 50 字符作为内容指纹

    Args:
        content:     chunk 文本内容
        document_id: 所属文档 UUID
        chunk_index: 在文档中的顺序位置（0-based）

    Returns:
        32 位十六进制 MD5 字符串，如 "a1b2c3d4e5f678901234567890abcdef"

    Example:
        >>> generate_chunk_id("IOC 容器是 Spring 的核心", "doc-001", 3)
        '7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c'
    """
    raw = f"{document_id}_{chunk_index}_{content[:50]}"
    return hashlib.md5(raw.encode()).hexdigest()


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                 KnowledgeBaseClient — Milvus 交互客户端                       ║
# ║                                                                            ║
# ║  职责：                                                                     ║
# ║    1. 封装 MilvusClient 连接（类级别单例，进程内复用）                        ║
# ║    2. 写入：upsert_chunks / delete_document_chunks                           ║
# ║    3. 检索：_hybrid_search（Dense + Sparse 混合检索）                        ║
# ║    4. 过滤：_build_filter（租户/课程安全过滤表达式）                          ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class KnowledgeBaseClient:
    """
    Milvus 向量数据库交互客户端。

    单 Collection 设计（knowledge_domain）：
      所有课程、所有租户的 chunk 存入同一集合
      → 通过 tenant_id / course_id 标量字段过滤实现隔离
      → 不需要每个课程/租户创建单独的 Collection

    连接管理（类级别单例）：
      _client 是类变量，整个进程共享一个 MilvusClient TCP 连接
      → 避免每次检索都重建连接的开销
      → _loaded 标记确保 load_collection 只执行一次

    用法示例（建库）：
        kb = KnowledgeBaseClient()
        kb.delete_document_chunks("doc-uuid-xxx")   # 先删旧版本
        kb.upsert_chunks([chunk1, chunk2, ...])     # 再写入新版本

    用法示例（检索）：
        kb = KnowledgeBaseClient()
        embedder = BGEMEmbedder.get_instance()
        dense, sparse = embedder.encode_query("用户问题")
        candidates = kb._hybrid_search(dense, sparse, top_k=10,
                                        filters=kb._build_filter("tenant_default"))
    """

    # ── 类级别属性（所有实例共享）────────────────────────────────

    _client: Optional[MilvusClient] = None
    # 类级别 MilvusClient 连接单例
    # None = 尚未建立连接；首次 __init__ 时创建并赋值

    _loaded: bool = False
    # 标记 collection 是否已加载到内存（load_collection 只需执行一次）

    # ── 检索参数 ─────────────────────────────────────────────────

    ANN_EF = 64
    # HNSW 索引搜索时的候选集大小（ef 参数）
    # 含义：搜索时考察 64 个最近邻候选节点
    # 越大精度越高但速度越慢，64 是精度/速度的经验平衡点
    # 参考：Milvus 官方推荐 ef ∈ [top_k, 256]，top_k 通常为 10

    # ═══════════════════════════════════════════════════════════════
    # 连接管理
    # ═══════════════════════════════════════════════════════════════

    def __init__(self):
        """
        初始化 Milvus 客户端连接。

        连接策略（懒加载）：
          - 首次创建 KnowledgeBaseClient 实例时建立 TCP 连接
          - 后续创建的实例复用同一个 _client（类变量共享）
          - load_collection 也只执行一次（_loaded 标记）

        容错：
          - init_milvus.py 未运行时 load_collection 会失败 → 静默忽略
          - 此时仍可调用写入方法，数据写入后 collection 会隐式加载
        """
        # ── 建立连接（仅首次）────────────────────────────────────
        if KnowledgeBaseClient._client is None:
            settings = get_settings()
            uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
            KnowledgeBaseClient._client = MilvusClient(uri=uri)
            logger.info("milvus.connected", uri=uri)

        # ── 加载 Collection 到内存（仅首次）───────────────────────
        # load_collection 将索引和数据加载到 Milvus 内存中，加速检索
        if not KnowledgeBaseClient._loaded:
            try:
                KnowledgeBaseClient._client.load_collection(COLLECTION_NAME)
            except Exception:
                pass  # init_milvus.py 尚未运行时忽略，后续写入会自动触发加载
            KnowledgeBaseClient._loaded = True

    # ═══════════════════════════════════════════════════════════════
    # 写入操作
    # ═══════════════════════════════════════════════════════════════

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        批量写入文档块（Upsert = Insert + Update）。

        工作原理：
          根据主键 id 判断：
            id 不存在 → INSERT 新记录
            id 已存在 → UPDATE 覆盖旧记录
          → 配合 generate_chunk_id() 的确定性，天然支持幂等建库

        数据转换流程：
          DocumentChunk 对象列表 → dict 列表 → MilvusClient.upsert()

        注意：
          - content 被截断到 4096 字符（Milvus VARCHAR 字段的最大长度）
          - 截断在 Python 侧完成，避免 Milvus 端报错

        Args:
            chunks: DocumentChunk 对象列表（由 build_knowledge_base.py 的
                    embed_chunks() 产出）

        Returns:
            实际写入的 chunk 数量
        """
        if not chunks:
            return 0

        data = [
            {
                "id":               c.id,
                "embedding":        c.embedding,              # 1024 维 float 列表
                "sparse_embedding": c.sparse_embedding,       # {token_id: weight}
                "content":          c.content[:4096],         # 截断到 VARCHAR 上限
                "chunk_index":      c.chunk_index,
                "document_id":      c.document_id,
                "course_id":        c.course_id,
                "tenant_id":        c.tenant_id,
                "source_name":      c.source_name,
                "chunk_type":       c.chunk_type,
                "version":          c.version,
                "updated_at":       c.updated_at,
            }
            for c in chunks
        ]

        self._client.upsert(collection_name=COLLECTION_NAME, data=data)
        logger.info("knowledge_base.chunks_upserted", count=len(chunks))
        return len(chunks)

    def delete_document_chunks(self, document_id: str) -> None:
        """
        删除指定文档的所有 chunk。

        使用场景：
          文档内容更新时，先删除旧版本的所有 chunk，再 upsert 新版本。
          这是实现"文档级幂等更新"的关键步骤。

        安全措施：
          对 document_id 做引号转义，防止 Milvus filter 表达式注入。
          Milvus filter 使用类 SQL 语法，用户输入的 " 需转义为 \"。

        Args:
            document_id: 要删除的文档 UUID
        """
        safe_id = document_id.replace('"', '\\"')
        self._client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'document_id == "{safe_id}"',
        )
        logger.info("knowledge_base.document_deleted", document_id=document_id)

    # ═══════════════════════════════════════════════════════════════
    # 检索操作
    # ═══════════════════════════════════════════════════════════════

    def _hybrid_search(
        self,
        query_embedding: list[float],
        query_sparse: dict,
        top_k: int,
        filters: Optional[str] = None,
    ) -> list[dict]:
        """
        混合检索：Dense + Sparse 两路召回 → WeightedRanker 加权融合。

        检索流程：
          1. 构造两个 AnnSearchRequest：
             - Dense 路：embedding 字段，COSINE 相似度，HNSW 索引
             - Sparse 路：sparse_embedding 字段，IP（内积）度量
          2. Milvus 服务端并行执行两路检索
          3. WeightedRanker(0.7, 0.3) 加权融合：
             final_score = 0.7 × dense_score + 0.3 × sparse_score
          4. 按融合分数降序排列，返回 Top-K

        为什么 0.7:0.3？
          - 教学场景中，同一概念有多种中文表述方式
          - Dense（语义）权重更高 → "意思差不多"的也能排前面
          - Sparse（关键词）保留 → 技术术语（IOC、DI、AOP）精确命中不会丢
          - 如果精确术语检索更重要，可调整为 0.5:0.5

        返回的 score 是什么？
          score 是 WeightedRanker 的加权融合排序信号，不是概率值。
          它是不同量纲分数（COSINE ∈ [-1,1] + IP ∈ [0,∞)）的加权和，
          只保证同一次查询内的相对排序有意义，不能跨查询比较。
          → 后续交给 BGE-Reranker 做精细重排序，输出真正的相关度分数。

        Args:
            query_embedding: Dense 查询向量（1024 维，来自 encode_query()[0]）
            query_sparse:    Sparse 查询向量（{token_id: weight}，来自 encode_query()[1]）
            top_k:           每路召回数量（融合后同样取 top_k）
            filters:         Milvus bool 过滤表达式，如 'tenant_id == "xxx"'
                             为 None 时不加过滤条件

        Returns:
            候选文档列表，每项包含：
              - content:  str    — chunk 文本内容
              - score:    float  — 加权融合分数（排序信号，非概率）
              - metadata: dict   — 含 source_name / chunk_type / course_id /
                                    document_id / chunk_index
        """
        try:
            # ── Dense 语义检索请求 ───────────────────────────────
            # COSINE 度量：BGE-M3 dense 向量 L2 归一化后，内积 = 余弦相似度
            # ef 参数：HNSW 搜索候选集大小，ANN_EF=64 是精度/速度平衡点
            dense_req = AnnSearchRequest(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": self.ANN_EF}},
                limit=top_k,
                expr=filters,
            )

            # ── Sparse 关键词检索请求 ─────────────────────────────
            # IP（内积）度量：BGE-M3 lexical_weights 的标准匹配方式
            # token 权重大的词命中时贡献更多分数
            sparse_req = AnnSearchRequest(
                data=[query_sparse],
                anns_field="sparse_embedding",
                param={"metric_type": "IP"},
                limit=top_k,
                expr=filters,
            )

            # ── 指定返回字段（减少网络传输）──────────────────────
            output_fields = [
                "content", "source_name", "chunk_type",
                "course_id", "document_id", "chunk_index",
            ]

            # ── 两路融合检索 ─────────────────────────────────────
            # WeightedRanker：服务端并行执行两路 ANN 检索，结果加权融合后排序
            results = self._client.hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=[dense_req, sparse_req],
                ranker=WeightedRanker(0.7, 0.3),
                limit=top_k,
                output_fields=output_fields,
            )

            # ── 解析结果 ─────────────────────────────────────────
            # hybrid_search 返回 list[list[dict]]
            # 外层列表对应 batch 中的每条查询（只有 1 条），内层列表是 Top-K 结果
            candidates = []
            for hit in results[0]:
                candidates.append({
                    "content":  hit["entity"].get("content") or "",
                    "score":    hit.get("distance") or 0.0,
                    "metadata": {
                        "source_name": hit["entity"].get("source_name") or "",
                        "chunk_type":  hit["entity"].get("chunk_type")  or "text",
                        "course_id":   hit["entity"].get("course_id")   or "",
                        "document_id": hit["entity"].get("document_id") or "",
                        "chunk_index": hit["entity"].get("chunk_index") or 0,
                    },
                })

            logger.info("knowledge_base.hybrid_search_done", candidates=len(candidates))
            return candidates

        except Exception as e:
            logger.error("knowledge_base.hybrid_search_failed", error=str(e))
            return []  # 检索失败返回空列表，不阻断上层业务（由 with_retry 兜底）

    # ═══════════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _build_filter(tenant_id: str, course_id: Optional[str] = None) -> str:
        """
        构建 Milvus bool 过滤表达式。

        Milvus filter 使用类 SQL 语法，支持 ==、!=、and、or 等运算符。
        所有检索请求都必须带 tenant_id 过滤，否则会跨租户泄露数据。

        安全措施：
          对 tenant_id / course_id 做引号转义，防止 filter 表达式注入攻击。
          虽然 Milvus 不像 SQL 那样容易被注入，但防御性转义仍是好习惯。

        Args:
            tenant_id: 租户 ID（必填，由 API 层从 JWT 中提取）
            course_id: 课程 ID（可选，不传则不按课程过滤）

        Returns:
            Milvus filter 表达式字符串

        Example:
            >>> _build_filter("tenant_01")
            'tenant_id == "tenant_01"'

            >>> _build_filter("tenant_01", "course-uuid-123")
            'tenant_id == "tenant_01" and course_id == "course-uuid-123"'
        """
        safe_tenant = tenant_id.replace('"', '\\"')
        expr = f'tenant_id == "{safe_tenant}"'
        if course_id:
            safe_course = course_id.replace('"', '\\"')
            expr += f' and course_id == "{safe_course}"'
        return expr

    @staticmethod
    def generate_chunk_id(content: str, document_id: str, chunk_index: int) -> str:
        """
        生成 chunk 唯一 ID（MD5）。

        作为类方法提供，方便外部不需要单独 import 模块级函数。
        实际逻辑与模块级 generate_chunk_id() 完全相同。

        Args:
            content:     chunk 文本内容
            document_id: 所属文档 UUID
            chunk_index: 在文档中的顺序位置

        Returns:
            32 位十六进制 MD5 字符串
        """
        raw = f"{document_id}_{chunk_index}_{content[:50]}"
        return hashlib.md5(raw.encode()).hexdigest()


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              快速自测                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    """
    快速验证知识库检索链路：
      BGE-M3 编码 → Hybrid 检索 → 打印 Top-3 结果

    前置条件：
      1. Milvus 服务已启动（docker compose up -d milvus）
      2. 已执行 python scripts/init_milvus.py 初始化集合
      3. 已执行 python scripts/build_knowledge_base.py 导入文档

    运行方式：
      python -m backend.core.knowledge_base
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from dotenv import load_dotenv
    load_dotenv(".env.local")

    # ── 测试查询 ─────────────────────────────────────────────────
    query = "项目架构是怎么样的？"
    print(f"🔍 查询：{query}\n")

    # ── BGE-M3 编码 ─────────────────────────────────────────────
    embedder = BGEMEmbedder.get_instance()
    dense, sparse = embedder.encode_query(query)
    print(f"dense 维度：{len(dense)}，sparse 词数：{len(sparse)}")

    # ── Hybrid 混合检索 ─────────────────────────────────────────
    kb = KnowledgeBaseClient()
    candidates = kb._hybrid_search(
        query_embedding=dense,
        query_sparse=sparse,
        top_k=10,
        filters=kb._build_filter("tenant_default"),
    )

    # ── 展示结果 ─────────────────────────────────────────────────
    print(f"召回候选数：{len(candidates)}")
    for i, c in enumerate(candidates[:3]):
        print(f"\n[{i+1}] score={c['score']:.4f}  来源：{c['metadata']['source_name']}")
        print(f"    {c['content'][:120]}...")
