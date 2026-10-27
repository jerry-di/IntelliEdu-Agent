"""
client - MCP Client
Agent 节点通过此模块调用 MCP Server，无需关心 JSON-RPC 细节

"""
import json
from typing import Any

import httpx

from backend.core.logger import get_logger

logger = get_logger(__name__)


async def call_mcp_tool(
    server_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    timeout: float = 30.0,
) -> Any:
    """
    调用 stateless MCP Server 的单个工具。

    stateless_http=True 的 Server 每次 POST 完全自包含，无需先发 initialize 握手。

    Args:
        server_url: MCP Server 基础 URL，例如 "http://localhost:8000/mcp/kb"
        tool_name:  工具名称，与 @mcp.tool() 注册名一致
        arguments:  工具参数字典
        timeout:    请求超时（秒）

    Returns:
        工具返回值（JSON 反序列化后的 Python 对象）

    Raises:
        httpx.HTTPStatusError:  Server 返回 4xx/5xx
        ValueError:             JSON-RPC 错误（工具内部异常）
        httpx.TimeoutException: 请求超时
    """
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "tools/call",
        "params": {
            "name":      tool_name,
            "arguments": arguments,
        },
    }

    # json_response=True 的 Server 要求客户端声明 Accept: application/json，
    # 否则 Server 返回 -32600 "Not Acceptable"
    headers = {
        "Content-Type": "application/json",
        "Accept":        "application/json",
    }

    # trust_env=False：不读系统代理，localhost 直连
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(f"{server_url}/mcp", json=payload, headers=headers)
        resp.raise_for_status()

    data = resp.json()

    # JSON-RPC 错误信封
    if "error" in data:
        raise ValueError(
            f"MCP tool '{tool_name}' error: {data['error'].get('message', str(data['error']))}"
        )

    # FastMCP 对 list[dict] 的序列化：每个 dict 一个 TextContent 条目
    # 必须遍历所有 content 条目，不能只取 content[0]
    content = data.get("result", {}).get("content", [])
    if not content:
        return []

    items = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = text
        if isinstance(parsed, list):
            return parsed      # 整个列表在一个 TextContent 里
        items.append(parsed)

    return items if items else content


async def list_mcp_tools(server_url: str, timeout: float = 10.0) -> list[dict]:
    """
    列出 MCP Server 提供的所有工具（调试 / 验证用）。

    Returns:
        工具列表，每项含 name / description / inputSchema
    """
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(f"{server_url}/mcp", json=payload, headers=headers)
        resp.raise_for_status()

    return resp.json().get("result", {}).get("tools", [])


# ── 模块自测 ──
if __name__ == "__main__":
    import asyncio, sys
    sys.path.insert(0, str(__file__).split("/backend/")[0])
    from dotenv import load_dotenv
    load_dotenv(".env.local")

    # ── 阶段一：知识库 MCP（需先启动 knowledge_base_server.py）──
    async def test_kb():
        base  = "http://localhost:8001"
        tools = await list_mcp_tools(base)
        print(f"[KB] 已注册工具：{[t['name'] for t in tools]}\n")
        results = await call_mcp_tool(
            server_url=base,
            tool_name="search_knowledge_base",
            arguments={"query": "商品聚合大模型的双塔召回怎么实现", "tenant_id": "tenant_default"},
        )
        print(f"[KB] 命中 {len(results)} 条")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] score={r['score']:.4f}  confidence={r['confidence']:.4f}")
            print(f"       来源：{r['source_name']}")
            print(f"       {r['content'][:80]}...\n")

    # ── 阶段二：Web 搜索 MCP（需先启动 web_search_server.py）──
    async def test_search():
        base  = "http://localhost:8002"
        tools = await list_mcp_tools(base)
        print(f"\n[Search] 已注册工具：{[t['name'] for t in tools]}\n")
        results = await call_mcp_tool(
            server_url=base,
            tool_name="web_search",
            arguments={"query": "BGE-M3 向量模型", "max_results": 3},
        )
        print(f"[Search] 搜索结果 {len(results)} 条")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r['title']}")
            print(f"       {r['url']}")
            print(f"       {r['snippet'][:80]}...\n")

    # ── 阶段三：集成测试（需先启动 FastAPI）──
    async def test_integrated():
        kb_results = await call_mcp_tool(
            server_url="http://localhost:8000/mcp/kb",
            tool_name="search_knowledge_base",
            arguments={"query": "Qwen VL微调的基本步骤是什么", "tenant_id": "tenant_default"},
        )
        print(f"\n[集成-KB] 命中 {len(kb_results)} 条")
        for i, r in enumerate(kb_results, 1):
            print(f"  [{i}] score={r['score']:.4f}  {r['source_name']}")

        search_results = await call_mcp_tool(
            server_url="http://localhost:8000/mcp/search",
            tool_name="web_search",
            arguments={"query": "BGE-M3 向量模型", "max_results": 3},
        )
        print(f"\n[集成-Search] 搜索结果 {len(search_results)} 条")
        for i, r in enumerate(search_results, 1):
            print(f"  [{i}] {r['title']}")

    # 按需取消注释对应阶段
    asyncio.run(test_kb())             # 阶段一（需先启动 knowledge_base_server.py）
    # asyncio.run(test_search())       # 阶段二（需先启动 web_search_server.py）
    # asyncio.run(test_integrated())   # 阶段三（需先启动 FastAPI）
