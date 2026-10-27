"""
build_knowledge_base - 导入知识库文档

"""
# scripts/build_knowledge_base.py（完整版）

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document

from backend.core.knowledge_base import BGEMEmbedder, KnowledgeBaseClient, DocumentChunk, generate_chunk_id
from backend.core.llm_factory import get_llm

# ── 常量 ─────────────────────────────────────────────────────

MAX_CONTEXT_CONCURRENCY = 5    # Contextual 上下文生成的最大并发 LLM 请求数

CONTEXTUAL_CHUNK_PROMPT = """\
<document>
{document_text}
</document>

以下是需要在整个文档中定位的 chunk：
<chunk>
{chunk_content}
</chunk>

请用一句简洁的中文，描述这段内容在整个文档中的位置和作用，以便改善检索效果。
只输出这一句描述，不要加任何前缀或标签。"""


# ── Step 1：读取文档（5.2 已实现）──────────────────────────
# load_document(file_path: str) -> list[Document]
# 在此文件内定义，参见 5.2 节完整代码
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """
    加载 PDF 文档，每页返回一个 Document。

    只提取文字层内容；图片/扫描件页面 page_content 为空字符串，
    不报错（在 5.3 分块时会过滤掉空页）。

    Args:
        file_path: PDF 文件的本地路径

    Returns:
        list[Document]，每个 Document 对应一页
        metadata 包含 source（文件路径）和 page（页码，从 0 开始）
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"  [PDF] 加载完成：{len(pages)} 页 ← {Path(file_path).name}")
    return pages


def load_markdown(file_path: str) -> list[Document]:
    """
    加载 Markdown 文档，整个文件作为一个 Document 返回。

    不在这里做标题切分——那是 5.3 分块步骤的工作。
    这里只负责把文件内容读进内存。

    Args:
        file_path: Markdown 文件的本地路径（.md 或 .markdown）

    Returns:
        list[Document]，只有一个元素，page_content 为文件全文
        metadata 包含 source（文件路径）
    """
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    char_count = len(docs[0].page_content)
    print(f"  [MD]  加载完成：{char_count} 字符 ← {Path(file_path).name}")
    return docs


def load_document(file_path: str) -> list[Document]:
    """
    统一文档加载入口。

    根据文件扩展名自动选择加载器：
        .pdf            → PyPDFLoader（文字层提取）
        .md / .markdown → TextLoader（纯文本读取）

    Args:
        file_path: 文档本地路径

    Returns:
        list[Document]

    Raises:
        ValueError: 不支持的文件类型
        FileNotFoundError: 文件不存在
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in (".md", ".markdown"):
        return load_markdown(file_path)
    else:
        raise ValueError(
            f"不支持的文件类型：{ext}\n"
            f"当前支持：.pdf / .md / .markdown\n"
            f"提示：可用 markitdown 将 Word/PPT 转换为 .md 后再导入"
        )

# ── Step 2：智能分块（5.3 已实现）──────────────────────────
# split_documents(docs: list[Document], file_path: str) -> list[Document]
# 在此文件内定义，参见 5.3 节完整代码
# scripts/build_knowledge_base.py（阶段版：文档加载 + 分块，5.4 / 5.5 继续补全）

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)

# ── 模块级分块器单例 ──────────────────────────────────────────
_MD_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#",   "H1"),
        ("##",  "H2"),
        ("###", "H3"),
        ("####", "H4"),
    ],
    strip_headers=False,
)

_CHAR_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)


# ── 文档加载（5.2 内容，此处合并为完整文件）────────────────────

def load_document(file_path: str) -> list[Document]:
    """统一文档加载入口，根据扩展名选择 Loader"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")
    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        print(f"  [PDF] 加载完成：{len(pages)} 页 ← {path.name}")
        return pages
    elif ext in (".md", ".markdown"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        print(f"  [MD]  加载完成：{len(docs[0].page_content)} 字符 ← {path.name}")
        return docs
    else:
        raise ValueError(
            f"不支持的文件类型：{ext}\n"
            f"当前支持：.pdf / .md / .markdown\n"
            f"提示：可用 markitdown 将 Word/PPT 转换为 .md 后再导入"
        )


# ── PDF 分块 ──────────────────────────────────────────────────

def split_pdf_documents(pages: list[Document]) -> list[Document]:
    """PDF 文档分块：过滤空页 + RecursiveCharacterTextSplitter"""
    non_empty_pages = [p for p in pages if len(p.page_content.strip()) > 20]
    skipped = len(pages) - len(non_empty_pages)
    if skipped > 0:
        print(f"  过滤空页：{skipped} 页（图片/扫描件页）")

    chunks = _CHAR_SPLITTER.split_documents(non_empty_pages)

    for chunk in chunks:
        filename = Path(chunk.metadata.get("source", "未知文件")).stem
        page_num = chunk.metadata.get("page", 0) + 1
        chunk.metadata["source_name"] = f"{filename} 第{page_num}页"

    print(f"  [PDF] 分块完成：{len(non_empty_pages)} 页 → {len(chunks)} 个 chunk")
    return chunks


# ── Markdown 分块 ─────────────────────────────────────────────

def split_markdown_documents(
    docs: list[Document],
    chunk_size: int = 1200,      # 代码类内容默认 1200，纯文字可调低到 600~800
    chunk_overlap: int = 100,
) -> list[Document]:
    """Markdown 文档分块：MarkdownHeaderTextSplitter + MarkdownTextSplitter 两阶段"""
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    header_chunks: list[Document] = []
    for doc in docs:
        sections = _MD_HEADER_SPLITTER.split_text(doc.page_content)
        source_path = doc.metadata.get("source", "")
        for section in sections:
            section.metadata["source"] = source_path
        header_chunks.extend(sections)

    final_chunks = splitter.split_documents(header_chunks)

    for chunk in final_chunks:
        source_path = chunk.metadata.get("source", "")
        filename    = Path(source_path).stem if source_path else "未知文件"
        parts = [
            chunk.metadata.get("H1", ""),
            chunk.metadata.get("H2", ""),
            chunk.metadata.get("H3", ""),
            chunk.metadata.get("H4", ""),
        ]
        parts = [p for p in parts if p]
        chunk.metadata["source_name"] = (
            f"{filename} > {' > '.join(parts)}" if parts else filename
        )

    print(f"  [MD]  分块完成：{len(docs)} 个文件 → {len(final_chunks)} 个 chunk")
    return final_chunks


# ── 统一分块入口 ──────────────────────────────────────────────

def split_documents(docs: list[Document], file_path: str) -> list[Document]:
    """统一分块入口，根据文件类型自动选择分块策略"""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return split_pdf_documents(docs)
    elif ext in (".md", ".markdown"):
        return split_markdown_documents(docs)
    else:
        raise ValueError(f"不支持的文件类型：{ext}")


# ── Step 3：BGE-M3 嵌入 ────────────────────────────────────────

def embed_chunks(
    chunks: list[Document],
    course_id: str,
    document_id: str,
    tenant_id: str = "tenant_default",
    version: str = "1.0",
) -> list[DocumentChunk]:
    """
    对分块后的文档列表做 BGE-M3 嵌入，生成 DocumentChunk 列表。

    一步完成：
      1. 提取每个 chunk 的文本内容
      2. 调用 BGE-M3 批量编码 → dense + sparse 向量
      3. 组装 DocumentChunk 数据对象
    """
    if not chunks:
        print("  ⚠️  无 chunk 需要嵌入，跳过")
        return []

    embedder = BGEMEmbedder.get_instance()

    # 提取文本列表
    texts = [c.page_content for c in chunks]

    # 批量编码
    print(f"  正在编码 {len(texts)} 个 chunk…")
    dense_vecs, sparse_vecs = embedder.encode(texts)

    # 组装 DocumentChunk 列表
    doc_chunks: list[DocumentChunk] = []
    now_ts = int(time.time())

    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        chunk_id = generate_chunk_id(content, document_id, i)

        # 检测 chunk 类型：简单启发式
        if content.strip().startswith("```"):
            chunk_type = "code"
        elif "|" in content and content.count("|") > 2:
            chunk_type = "table"
        else:
            chunk_type = "text"

        doc_chunks.append(DocumentChunk(
            id=chunk_id,
            content=content,
            embedding=dense_vecs[i],
            sparse_embedding=sparse_vecs[i],
            course_id=course_id,
            document_id=document_id,
            source_name=chunk.metadata.get("source_name", "未知来源"),
            chunk_index=i,
            version=version,
            tenant_id=tenant_id,
            chunk_type=chunk_type,
            updated_at=now_ts,
        ))

    print(f"  嵌入完成：{len(doc_chunks)} 个 chunk → {len(dense_vecs[0])} 维稠密向量")
    return doc_chunks


import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)
backend_path = os.path.dirname(os.path.dirname(__file__))


# ──────────────────────────────────────────────────────────────
# BGE-M3 本地嵌入模型（进程内单例，dense + sparse 双输出）
# ──────────────────────────────────────────────────────────────

class BGEMEmbedder:
    """
    BGE-M3 本地嵌入模型单例。

    一次推理同时输出：
      - dense 向量（1024 维浮点数组，用于语义相似度检索）
      - sparse 向量（{token_id: weight} 字典，用于关键词精确检索）

    进程内单例：首次调用 get_instance() 时加载模型（约5-15秒），
    后续调用直接返回同一实例，不重复加载。

    用法：
        embedder = BGEMEmbedder.get_instance()
        dense, sparse = embedder.encode_query("什么是 Spring IOC？")
    """

    _instance: Optional["BGEMEmbedder"] = None   # 单例持有

    def __init__(self, model_path: str):
        # ── 兼容性补丁1：FlagEmbedding 1.3.x 依赖 transformers 内部函数 ──
        # transformers>=5.0 移除了 is_torch_fx_available，
        # 但当前锁定 transformers==4.51.0 不受影响。
        # 此补丁作为保险，避免未来升级时报 ImportError。
        import importlib.util as _ilu
        from transformers.utils import import_utils as _tf_iu
        if not hasattr(_tf_iu, "is_torch_fx_available"):
            _tf_iu.is_torch_fx_available = (
                lambda: _ilu.find_spec("torch.fx") is not None
            )

        # ── 兼容性补丁2：修复 XLMRobertaModel 不接受 dtype 参数的问题 ──
        # 某些 transformers 版本中 XLMRobertaModel.__init__() 不接受 dtype 关键字参数，
        # 但 FlagEmbedding 内部会传入该参数，导致 TypeError。
        # 通过子类覆盖，在调用父类前丢弃 dtype 参数。
        from transformers.models.xlm_roberta import modeling_xlm_roberta as _xlm
        _OriginalXLMRoberta = _xlm.XLMRobertaModel

        class _PatchedXLMRobertaModel(_OriginalXLMRoberta):
            def __init__(self, config, **kwargs):
                kwargs.pop("dtype", None)  # 丢弃 FlagEmbedding 传入的 dtype
                super().__init__(config, **kwargs)

        _xlm.XLMRobertaModel = _PatchedXLMRobertaModel

        import torch
        from FlagEmbedding import BGEM3FlagModel

        logger.info("bge_m3.loading", model_path=model_path)

        # ── fp16 仅在 CUDA 上启用，MPS（Apple M系列）不启用 ──
        # MPS 在 BGE-M3 attention 矩阵乘法上会触发 LLVM ERROR，
        # CPU 模式下用 fp32，速度稍慢但稳定。
        _use_fp16 = torch.cuda.is_available()

        self._model = BGEM3FlagModel(
            model_name_or_path=model_path,
            use_fp16=_use_fp16,
        )
        logger.info("bge_m3.loaded", use_fp16=_use_fp16)

    @classmethod
    def get_instance(cls) -> "BGEMEmbedder":
        """获取单例（首次调用时加载模型，后续复用）"""
        if cls._instance is None:
            bge3_path = os.path.join(backend_path, get_settings().bge_m3_model_path)
            cls._instance = BGEMEmbedder(bge3_path)
        return cls._instance

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
    ) -> tuple[list[list[float]], list[dict]]:
        """
        批量编码文本，同时返回 dense 和 sparse 两种向量。

        Args:
            texts:      待编码的文本列表
            batch_size: 单次推理批大小，越大速度越快但显存占用越多；
                        12 是 16GB 显存 / 统一内存下的经验值

        Returns:
            (dense_vecs, sparse_vecs)
              dense_vecs:  list of 1024-dim float 向量，每项对应 texts[i]
              sparse_vecs: list of {token_id: weight} 字典，每项对应 texts[i]
        """
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,            # BGE-M3 支持最长 8192 token，覆盖大多数 chunk
            return_dense=True,          # 输出稠密语义向量
            return_sparse=True,         # 输出稀疏关键词向量
            return_colbert_vecs=False,  # ColBERT 多向量表示，本项目不用
        )

        dense_vecs = output["dense_vecs"].tolist()   # numpy → Python list

        # sparse: numpy.float16 → Python float
        # 必须转换！LangGraph MemorySaver 用 msgpack 序列化 State，
        # msgpack 不支持 numpy.float16，会在运行时抛 TypeError。
        sparse_vecs = [
            {int(k): float(v) for k, v in d.items()}
            for d in output["lexical_weights"]
        ]

        return dense_vecs, sparse_vecs

    def encode_query(self, text: str) -> tuple[list[float], dict]:
        """
        编码单条查询，返回 (dense_vec, sparse_vec)。

        查询时调用此方法（而非 encode），batch_size=1 避免不必要的 padding。

        Returns:
            (dense_vec, sparse_vec)
              dense_vec:  1024-dim float 列表
              sparse_vec: {token_id: weight} 字典
        """
        dense_list, sparse_list = self.encode([text], batch_size=1)
        return dense_list[0], sparse_list[0]


# ──────────────────────────────────────────────────────────────
# 数据类：DocumentChunk（建库写入 Milvus 的数据结构）
# ──────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """
    准备写入 Milvus 的单个文档块，字段与 Milvus Schema 一一对应。

    id:               全局唯一 ID（MD5 of content + document_id + chunk_index）
    content:          chunk 文本（Contextual RAG 模式下含 LLM 生成的上下文描述前缀）
    embedding:        Dense 向量（BGE-M3，1024 维）
    sparse_embedding: Sparse 向量（{token_id: weight}，BGE-M3 lexical weights）
    source_name:      来源标注（检索结果展示用，如 "Java讲义 > 第3章 > 3.1 IOC"）
    """
    id:               str
    content:          str
    embedding:        list[float]
    sparse_embedding: dict
    course_id:        str
    document_id:      str
    source_name:      str
    chunk_type:       str                  # "text" / "code" / "table"
    chunk_index:      int
    version:          str
    tenant_id:        str = "tenant_default"
    updated_at:       int = field(default_factory=lambda: int(time.time()))


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def generate_chunk_id(content: str, document_id: str, chunk_index: int) -> str:
    """
    生成 chunk 全局唯一 ID（MD5 散列）。

    用 document_id + chunk_index + content 前缀组合，确保：
    - 同一文档不同位置的 chunk 不冲突
    - 内容不变时 ID 稳定（幂等重建时不会重复插入）

    注意：此函数后续会作为 KnowledgeBaseClient 的静态方法重新出现（5.6），
    届时 build_knowledge_base.py 会改为调用 KnowledgeBaseClient.generate_chunk_id()。
    """
    raw = f"{document_id}_{chunk_index}_{content[:50]}"
    return hashlib.md5(raw.encode()).hexdigest()

# ── Step 2.5：Contextual RAG 上下文增强 ─────────────────────

async def generate_chunk_context(
    llm,
    document_text: str,
    chunk_content: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """
    用 LLM 为单个 chunk 生成一句定位描述。

    失败时返回空字符串，调用方保留原始 chunk 文本（降级处理）。

    Args:
        llm:           DeepSeek LLM 实例（via get_llm）
        document_text: 整篇文档全文（截断至 8000 字）
        chunk_content: 当前 chunk 的原始文本
        semaphore:     并发限流（最多 MAX_CONTEXT_CONCURRENCY 个 LLM 请求同时进行）
    """
    async with semaphore:
        try:
            from langchain_core.messages import HumanMessage
            prompt = CONTEXTUAL_CHUNK_PROMPT.format(
                document_text=document_text,
                chunk_content=chunk_content,
            )
            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            ctx = (
                resp.text
                if hasattr(resp, "text") and not callable(resp.text)
                else str(resp.content)
            ).strip()
            return ctx
        except Exception as e:
            print(f"   [warning] 上下文生成失败，保留原始 chunk：{e}")
            return ""


async def add_context(
    chunks: list[Document],
    docs: list[Document],
    concurrency: int = MAX_CONTEXT_CONCURRENCY,
) -> list[Document]:
    """
    Contextual RAG：并发为所有 chunk 生成上下文描述，拼接到 chunk 文本前方。

    拼接后格式：
        "<上下文描述一句话>\\n\\n<原始 chunk 文本>"

    拼接后再做嵌入（embed_chunks），向量同时编码"在哪里"和"说了什么"两层信息。

    Args:
        chunks:      split_documents() 输出的 list[Document]
        docs:        load_document() 输出的原始 list[Document]（用于构建全文参考）
        concurrency: 最大并发 LLM 请求数（默认 5，防止触发 API 限流）

    Returns:
        page_content 已被就地修改（拼接上下文）的 list[Document]
    """
    # 拼接全文供 LLM 参考（截断 8000 字，避免超出模型 context 长度）
    full_doc_text = "\n\n".join(d.page_content for d in docs)[:8000]

    llm       = get_llm("qa", temperature=0)
    semaphore = asyncio.Semaphore(concurrency)

    # 并发调用 LLM，为每个 chunk 生成上下文描述
    contexts = await asyncio.gather(*[
        generate_chunk_context(llm, full_doc_text, c.page_content, semaphore)
        for c in chunks
    ])

    enriched = 0
    for chunk, ctx in zip(chunks, contexts):
        if ctx:
            chunk.page_content = f"{ctx}\n\n{chunk.page_content}"
            enriched += 1

    print(f"  上下文增强完成：{enriched}/{len(chunks)} 个 chunk 已添加描述")
    return chunks


# ── Step 4：写入 Milvus ────────────────────────────────────────

def write_to_milvus(doc_chunks: list[DocumentChunk]) -> None:
    """
    将 embed_chunks() 产出的 DocumentChunk 列表写入 Milvus。

    先按 document_id 删除同文档旧版本 chunk，再批量 upsert，
    保证文档更新时不残留旧数据。
    """
    if not doc_chunks:
        print("  ⚠️  无 chunk 可写入，跳过")
        return

    kb          = KnowledgeBaseClient()
    document_id = doc_chunks[0].document_id

    print(f"  🗑️  删除旧版本 chunk（document_id={document_id[:8]}…）")
    kb.delete_document_chunks(document_id)

    written = kb.upsert_chunks(doc_chunks)
    print(f"  ✅ 写入完成：{written} 个 chunk → knowledge_domain")


# ── 主流水线 ─────────────────────────────────────────────────

async def build_pipeline(
    file_path:   str,
    course_id:   str,
    document_id: str,
    tenant_id:   str = "tenant_default",
    version:     str = "1.0",
    use_context: bool = True,
) -> None:
    """
    知识库建库完整流水线（五步）：

      Step 1   读取文档（PyPDFLoader / TextLoader）
      Step 2   智能分块（MarkdownHeaderTextSplitter / RecursiveCharacterTextSplitter）
      Step 2.5 Contextual RAG 上下文增强（LLM 并发，可跳过）
      Step 3   BGE-M3 嵌入（dense + sparse 双向量）
      Step 4   写入 Milvus（MilvusClient upsert）
    """
    print(f"\n{'='*55}")
    print(f" EduAgent 知识库构建")
    print(f" 文件      ：{file_path}")
    print(f" 课程      ：{course_id}")
    print(f" 文档 ID   ：{document_id}")
    print(f" 租户      ：{tenant_id}")
    print(f" Contextual RAG：{'启用' if use_context else '跳过（--no-context）'}")
    print(f"{'='*55}\n")

    # Step 1：读取
    print("📖 Step 1/4  读取文档…")
    docs = load_document(file_path)

    # Step 2：分块
    print("\n✂️  Step 2/4  智能分块…")
    chunks = split_documents(docs, file_path)

    # Step 2.5：Contextual RAG（可选）
    if use_context and chunks:
        print(f"\n🧠 Step 2.5  Contextual RAG 上下文增强"
              f"（并发={MAX_CONTEXT_CONCURRENCY}）…")
        chunks = await add_context(chunks, docs)

    # Step 3：嵌入
    print("\n🔢 Step 3/4  BGE-M3 嵌入…")
    doc_chunks = embed_chunks(
        chunks,
        course_id=course_id,
        document_id=document_id,
        tenant_id=tenant_id,
        version=version,
    )

    # Step 4：写入
    print("\n💾 Step 4/4  写入 Milvus…")
    write_to_milvus(doc_chunks)

    print(f"\n🎉 完成！共处理 {len(doc_chunks)} 个 chunk")
    print(f"   document_id = {document_id}")
    print(f"   ⚠️  更新此文档时请保留此 document_id")


# ── CLI 入口 ─────────────────────────────────────────────────
# 直接修改下方变量值，然后运行：python scripts/build_knowledge_base.py

if __name__ == "__main__":
    _PROJECT_ROOT = Path(__file__).parent.parent
    FILE_PATH   = str(_PROJECT_ROOT / "sample" / "sample2.md")
    COURSE_ID   = "3e76aeed-5e01-4aa7-be8d-2055d12b9ea7"   # 替换为实际课程 UUID
    DOCUMENT_ID = None          # None = 自动生成；更新同一文档时填入上次输出的 ID
    TENANT_ID   = "tenant_default"
    VERSION     = "1.0"
    USE_CONTEXT = True          # False = 跳过 Contextual RAG（快速调试，不消耗 API 配额）

    doc_id = DOCUMENT_ID or str(uuid.uuid4())

    asyncio.run(build_pipeline(
        file_path=FILE_PATH,
        course_id=COURSE_ID,
        document_id=doc_id,
        tenant_id=TENANT_ID,
        version=VERSION,
        use_context=USE_CONTEXT,
    ))





