# backend/agents/exam/graph.py

from langgraph.graph import StateGraph, START, END

from backend.agents.exam.state import ExamState
from backend.agents.exam.nodes import (
    parse_word_node,
    load_questions_meta_node,
    run_three_tracks_node,
    aggregate_results_node,
    analyze_weak_points_node,
    notify_teacher_node,
    teacher_review_node,
    apply_teacher_decision_node,
    publish_results_node,
)
from backend.core.memory import get_memory_saver


def build_exam_graph():
    """
    构建并编译试卷批改 Agent 的 LangGraph 状态图。

    执行链路（线性）：
        parse_word → load_questions_meta → run_three_tracks
        → aggregate_results → analyze_weak_points
        → notify_teacher → teacher_review [interrupt]
        → apply_teacher_decision → publish_results → END
    """
    builder = StateGraph(ExamState)

    # ── 注册节点 ──────────────────────────────────────────────
    builder.add_node("parse_word",             parse_word_node)
    builder.add_node("load_questions_meta",    load_questions_meta_node)
    builder.add_node("run_three_tracks",       run_three_tracks_node)
    builder.add_node("aggregate_results",      aggregate_results_node)
    builder.add_node("analyze_weak_points",    analyze_weak_points_node)
    builder.add_node("notify_teacher",         notify_teacher_node)
    builder.add_node("teacher_review",         teacher_review_node)
    builder.add_node("apply_teacher_decision", apply_teacher_decision_node)
    builder.add_node("publish_results",        publish_results_node)

    # ── 固定边（线性链）──────────────────────────────────────
    builder.add_edge(START,                    "parse_word")
    builder.add_edge("parse_word",             "load_questions_meta")
    builder.add_edge("load_questions_meta",    "run_three_tracks")
    builder.add_edge("run_three_tracks",       "aggregate_results")
    builder.add_edge("aggregate_results",      "analyze_weak_points")
    builder.add_edge("analyze_weak_points",    "notify_teacher")
    builder.add_edge("notify_teacher",         "teacher_review")
    builder.add_edge("teacher_review",         "apply_teacher_decision")
    builder.add_edge("apply_teacher_decision", "publish_results")
    builder.add_edge("publish_results",        END)

    # ── 编译，绑定 MemorySaver ────────────────────────────────
    # interrupt() 暂停后 State 必须持久化，否则无法恢复执行
    checkpointer = get_memory_saver("exam")
    return builder.compile(checkpointer=checkpointer)


# ════════════════════════════════════════════════════════════════
# 模块自测：完整图端到端测试（含 HitL interrupt 暂停与恢复）
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, asyncio, uuid
    sys.path.insert(0, ".")

    from dotenv import load_dotenv
    load_dotenv(".env.local")

    from sqlalchemy import text
    from langgraph.types import Command
    from backend.dependencies import AsyncSessionLocal

    graph    = build_exam_graph()
    EXAM_ID  = "e0000001-0000-0000-0000-000000000001"

    async def main():
        # ── 测试准备：获取真实 student UUID，写入临时提交记录 ──
        async with AsyncSessionLocal() as session:
            row = await session.execute(
                text("SELECT id FROM users WHERE username = 'student01' LIMIT 1")
            )
            student_uuid = str(row.scalar())

        submission_id = str(uuid.uuid4())

        async with AsyncSessionLocal() as session:
            async with session.begin():
                # 清除旧测试记录
                await session.execute(
                    text("""
                        DELETE FROM exam_reviews
                        WHERE submission_id IN (
                            SELECT id FROM exam_submissions
                            WHERE exam_id = :exam_id AND student_id = :student_id
                        )
                    """),
                    {"exam_id": EXAM_ID, "student_id": student_uuid},
                )
                await session.execute(
                    text("DELETE FROM exam_submissions WHERE exam_id=:exam_id AND student_id=:student_id"),
                    {"exam_id": EXAM_ID, "student_id": student_uuid},
                )
                await session.execute(
                    text("""
                        INSERT INTO exam_submissions (id, exam_id, student_id, tenant_id, status)
                        VALUES (:id, :exam_id, :student_id, 'tenant_default', 'processing')
                    """),
                    {"id": submission_id, "exam_id": EXAM_ID, "student_id": student_uuid},
                )

        thread_id = f"test-{uuid.uuid4()}"
        config    = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "messages":        [],
            "word_file_path":  "./backend/agents/exam/student_answer.docx",
            "exam_id":         EXAM_ID,
            "submission_id":   submission_id,
            "student_id":      student_uuid,
            "tenant_id":       "tenant_default",
            "session_id":      "test-session",
            "teacher_decision": None,
            "fallback_used":   False,
            "sandbox_skipped": False,
        }

        # ══ 第一次 ainvoke：跑到 teacher_review_node 的 interrupt() 自动暂停 ══
        print("=" * 60)
        print("第一次 ainvoke：启动图，等待 interrupt 暂停...")
        print("=" * 60)
        result1 = await graph.ainvoke(initial_state, config=config)

        summary = result1.get("pre_review_summary", {})
        print(f"\n图已暂停（interrupt 触发）")
        print(f"AI 预评总分：{summary.get('total_score')} / {summary.get('full_score')}")
        print(f"需复核题数：{summary.get('needs_review_count')}")
        print(f"薄弱点数量：{len(result1.get('weak_points', []))}")
        print(f"\n>>> 此刻图冻结，等待教师决策 <<<\n")

        # ══ 模拟教师操作：approve，第二次 ainvoke 恢复图 ══════
        decision = {
            "action":        "approve",
            "modifications": [],
            "teacher_id":    str(uuid.uuid4()),
        }

        print("=" * 60)
        print("第二次 ainvoke：Command(resume=decision)，恢复图执行...")
        print("=" * 60)
        result2 = await graph.ainvoke(Command(resume=decision), config=config)

        final = result2.get("final_results", [])
        print(f"\n图执行完毕")
        print(f"最终批改题数：{len(final)}")
        for q in final:
            print(f"  题{q['question_no']} [{q['question_type']}] "
                  f"最终得分 {q['final_score']}/{q['full_score']}")

    asyncio.run(main())
