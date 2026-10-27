"""
knowledge_base_server - 知识库 MCP Server
封装 retrieve() Pipeline，让外部 MCP Client 能搜索 EduAgent 知识库

"""
import asyncio
import os
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="EduAgent-KnowledgeBase",
    stateless_http=True,           # 无 session，每次 POST 自包含
    json_response=True,            # 返回值自动序列化为 JSON 字符串
)


@mcp.tool()
async def search_knowledge_base(
    query: str,
    tenant_id: str,
    course_id: str | None = None,
    top_k: int = 3,
) -> list[dict]:
    """
    Hybrid semantic + keyword search over the EduAgent Milvus knowledge base.

    Uses BGE-M3 dual-vector encoding combined with Milvus hybrid_search
    and WeightedRanker fusion, followed by BGE-Reranker cross-encoder reranking.

    Args:
        query:     Search query in Chinese or English
        tenant_id: Tenant identifier for data isolation (e.g. "tenant_default")
        course_id: Optional course UUID to restrict search scope
        top_k:     Number of documents to return after reranking (default: 3)

    Returns:
        List of ranked documents, each with:
        - content:            chunk text
        - source_name:        document name and chapter
        - score:              reranker confidence score [0, 1]
        - confidence:         top document confidence score
        - is_high_confidence: whether confidence >= 0.75
    """
    from backend.core.reranker import retrieve
    from backend.core.logger import get_logger

    logger = get_logger(__name__)

    # retrieve() 内部包含 BGE-M3 编码（CPU-bound）和 CrossEncoder 推理（CPU-bound），
    # 用 run_in_executor 丢到线程池，避免阻塞 async 事件循环
    loop = asyncio.get_running_loop()
    ranked_docs, confidence = await loop.run_in_executor(
        None,
        lambda: retrieve(
            query=query,
            tenant_id=tenant_id,
            course_id=course_id,
            rerank_top_k=top_k,
        ),
    )

    logger.info(
        "kb_mcp.search_done",
        query_preview=query[:50],
        hits=len(ranked_docs),
        confidence=round(confidence, 4),
    )

    return [
        {
            "content":            doc.content,
            "source_name":        doc.metadata.get("source_name", ""),
            "score":              round(doc.score, 6),
            "confidence":         round(confidence, 4),
            "is_high_confidence": confidence >= 0.75,
        }
        for doc in ranked_docs
    ]


# ── 独立运行入口 ──
if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    sys.path.insert(0, str(__file__).split("/backend/")[0])
    load_dotenv(".env.local")

    port = int(os.getenv("KB_MCP_SERVER_PORT", "8001"))
    print(f"Knowledge Base MCP Server -> http://localhost:{port}/mcp")
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=port)
