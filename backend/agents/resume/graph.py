"""
graph - 简历审查 LangGraph 工作流
将 8 个节点串成 DAG：①②→③→④→⑤→⑥→⑦→⑧

"""

from langgraph.graph import StateGraph, END   # StateGraph: 状态图构建器 / END: 终止标记

from backend.agents.resume.state import ResumeState              # 工作流全局状态类型（TypedDict）
from backend.agents.resume.nodes import (                        # 全部 8 个节点函数
    upload_to_minio_node,        # ① 上传 MinIO（本地模式空跑，预留扩展点）
    download_pdf_node,           # ② 下载 PDF（本地模式空跑，预留扩展点）
    extract_text_node,           # ③ PDF → 纯文本（PyMuPDF 解析 + 双栏适配）
    extract_structured_node,     # ④ LLM 结构化提取（ResumeStructured）
    run_six_dimensions_node,     # ⑤ 六维度并行评审（asyncio.gather 同时评 6 维）
    diagnose_issues_node,        # ⑥ 问题诊断（Think 推理 → IssueList 排序）
    generate_summary_node,       # ⑦ 整体评价（ResumeSummary）
    save_results_node,           # ⑧ 持久化结果到 PG + 清理临时 PDF
)


def build_resume_graph() -> StateGraph:
    """
    构建简历审查工作流并编译为可执行图。

    整体流程（线性串行，无分支/条件边）：
        start → ① upload → ② download → ③ extract_text
              → ④ extract_structured → ⑤ run_six_dimensions
              → ⑥ diagnose_issues → ⑦ generate_summary
              → ⑧ save_results → END

    为什么是线性而不是 DAG？
      - 每个节点依赖前一节点的输出（③ 需要 PDF 路径 → ④ 需要原文 → ⑤ 需要结构...）
      - 唯一的并行发生在节点⑤内部（6 个维度 asyncio.gather），而非图层面
      - 将来如果有不依赖彼此的节点（如结构化提取和 OCR），可改为并行边

    Returns:
        编译后的 StateGraph 实例，调用 .ainvoke(state_dict) 即可执行
    """
    # ── ① 创建状态图，绑定 ResumeState 类型 ──
    # StateGraph 会校验每个节点返回的 dict 是否与 ResumeState 字段兼容
    builder = StateGraph(ResumeState)

    # ── ② 注册全部 8 个节点 ──
    # add_node(名称, 函数)：名称用于 add_edge 串联，也出现在 LangGraph 调试日志中
    builder.add_node("upload_to_minio",    upload_to_minio_node)     # ① 上传（空跑）
    builder.add_node("download_pdf",       download_pdf_node)        # ② 下载（空跑）
    builder.add_node("extract_text",       extract_text_node)        # ③ PDF 解析
    builder.add_node("extract_structured", extract_structured_node)  # ④ 结构化提取
    builder.add_node("run_six_dimensions", run_six_dimensions_node)  # ⑤ 六维度评审
    builder.add_node("diagnose_issues",    diagnose_issues_node)     # ⑥ 问题诊断
    builder.add_node("generate_summary",   generate_summary_node)    # ⑦ 整体评价
    builder.add_node("save_results",       save_results_node)        # ⑧ 持久化

    # ── ③ 设置入口 ──
    # 工作流从 ① 开始（上传接口调用后，PDF 已在本地磁盘）
    builder.set_entry_point("upload_to_minio")

    # ── ④ 串联边（线性流水线）──
    # add_edge(A, B)：A 完成后自动执行 B，A 的返回 dict 会合并到 State
    builder.add_edge("upload_to_minio",    "download_pdf")           # ① → ②
    builder.add_edge("download_pdf",       "extract_text")           # ② → ③
    builder.add_edge("extract_text",       "extract_structured")     # ③ → ④
    builder.add_edge("extract_structured", "run_six_dimensions")     # ④ → ⑤
    builder.add_edge("run_six_dimensions", "diagnose_issues")        # ⑤ → ⑥
    builder.add_edge("diagnose_issues",    "generate_summary")       # ⑥ → ⑦
    builder.add_edge("generate_summary",   "save_results")           # ⑦ → ⑧
    builder.add_edge("save_results",       END)                      # ⑧ → 终止

    # ── ⑤ 编译：校验图完整性 + 包装成可执行对象 ──
    return builder.compile()
    # compile() 会检查：
    #   - 是否有孤立节点（没有入边也没有出边）
    #   - 是否所有路径都有明确终点
    #   - 节点签名是否与 State 兼容


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                         模块自测：验证图结构                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    g = build_resume_graph()                                       # 编译工作流
    # get_graph() 返回图的静态结构（nx.DiGraph），用于可视化/调试
    nodes = [n for n in g.get_graph().nodes if n not in ("__start__", "__end__")]
    # 过滤掉 LangGraph 自动添加的虚拟起止节点

    print("图编译成功:", g is not None)     # 验证编译未抛异常
    print("节点数:", len(nodes))           # 应该为 8
    print("节点:", nodes)                  # 按注册顺序列出
