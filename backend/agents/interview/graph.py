"""
graph - 模拟面试 Agent LangGraph 状态图
含条件分支：check_stage 决定走正常对话还是报告生成路径

"""
from langgraph.graph import StateGraph, START, END

from backend.agents.interview.state import InterviewState, InterviewStage
from backend.agents.interview.nodes import (
    load_context_node,
    check_stage_node,
    evaluate_answer_node,
    generate_response_node,
    generate_report_node,
    save_report_node,
    save_memory_node,
)
from backend.core.memory import get_memory_saver


def _route_after_check_stage(state: InterviewState) -> str:
    """
    check_stage 后的条件路由：
        - 已进入 FINISHED → 生成评估报告路径
        - 其他阶段 → 正常对话路径
    """
    if state.get("current_stage") == InterviewStage.FINISHED.value:
        return "generate_report"
    return "evaluate_answer"


def build_interview_graph():
    """
    构建并编译模拟面试 Agent 的 LangGraph 状态图。

    图拓扑：
        START → load_context → check_stage
                                   ├─(finished)→ generate_report → save_report → save_memory → END
                                   └─(else)    → evaluate_answer → generate_response → save_memory → END
    """
    builder = StateGraph(InterviewState)

    # 注册节点
    builder.add_node("load_context",      load_context_node)
    builder.add_node("check_stage",       check_stage_node)
    builder.add_node("evaluate_answer",   evaluate_answer_node)
    builder.add_node("generate_response", generate_response_node)
    builder.add_node("generate_report",   generate_report_node)
    builder.add_node("save_report",       save_report_node)
    builder.add_node("save_memory",       save_memory_node)

    # 固定边
    builder.add_edge(START,                "load_context")
    builder.add_edge("load_context",       "check_stage")

    # 条件分支
    builder.add_conditional_edges(
        "check_stage",
        _route_after_check_stage,
        {
            "generate_report": "generate_report",
            "evaluate_answer": "evaluate_answer",
        },
    )

    # 正常对话路径
    builder.add_edge("evaluate_answer",   "generate_response")
    builder.add_edge("generate_response", "save_memory")
    builder.add_edge("save_memory",       END)

    # 报告生成路径
    builder.add_edge("generate_report",   "save_report")
    builder.add_edge("save_report",       "save_memory")

    # 有 checkpointer：多轮对话需要状态跨请求持久化
    checkpointer = get_memory_saver("interview")
    return builder.compile(checkpointer=checkpointer)


# ════════════════════════════════════════════════════════════════
# 模块自测：验证图结构
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    g = build_interview_graph()
    nodes = [n for n in g.get_graph().nodes if n not in ("__start__", "__end__")]
    print(f"图编译成功: {g is not None}")
    print(f"节点数: {len(nodes)}")
    print(f"节点: {nodes}")
