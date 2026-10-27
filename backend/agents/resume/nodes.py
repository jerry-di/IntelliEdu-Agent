"""
nodes - 简历审查 Agent 全部 8 个节点定义
① upload_to_minio   ② download_pdf    ③ extract_text      ④ extract_structured
⑤ run_six_dimensions  ⑥ diagnose_issues  ⑦ generate_summary  ⑧ save_results

"""
# backend/agents/resume/nodes.py

import asyncio                         # 异步 I/O：并行 LLM 请求 + sleep 重试间隔
import json                            # 序列化 dict → JSON 字符串，写数据库 JSONB 列
import os                              # 清理本地临时 PDF 文件

from langchain_core.messages import HumanMessage, SystemMessage  # LangChain 消息：系统角色 + 用户输入
from sqlalchemy import text            # 写原生 SQL（UPDATE resume_reviews ...）

from backend.agents.resume.state import (
    ResumeState, ResumeStructured, DimensionScore, IssueList, ResumeSummary,
)                                      # Pydantic 数据模型：State / 结构化简历 / 评分 / 诊断 / 评价
from backend.agents.resume.prompts import (
    SYSTEM_PROMPT, EXTRACT_STRUCTURED_PROMPT, DIMENSION_REVIEW_PROMPTS,
    DIAGNOSE_ISSUES_PROMPT, GENERATE_SUMMARY_PROMPT, DIAGNOSE_THINK_PROMPT,
)                                      # 全部提示词：系统角色 / 结构化提取 / 六维度评审 / 诊断 / 评价 / Think
from backend.core.llm_factory import get_structured_llm, get_llm  # 模型工厂：普通模型 / 绑结构化 Schema 的模型
from backend.core.logger import get_logger                       # 结构化日志
from backend.dependencies import AsyncSessionLocal               # SQLAlchemy 异步会话（连接池）

logger = get_logger(__name__)           # 本模块日志器


# ════════════════════════════════════════════════════════════════
# 六维度配置表
# key     → 查 DIMENSION_REVIEW_PROMPTS 字典取提示词模板
# name    → 日志 / 前端展示
# weight  → 加权算总分（之和 = 1.0）
# focus   → 发给 LLM 的评审侧重点
# ════════════════════════════════════════════════════════════════
SIX_DIMENSIONS = [
    {"key": "project_depth",  "name": "项目深度",   "weight": 0.30,
     "focus": "项目描述是否有量化数据、技术选型理由、个人贡献、难点解决"},
    {"key": "tech_match",     "name": "技术匹配度", "weight": 0.25,
     "focus": "技术栈是否与目标岗位匹配，技能描述是否有层次（熟练/了解/掌握）"},
    {"key": "expression",     "name": "表达规范性", "weight": 0.15,
     "focus": "动词开头、STAR 结构、无错别字、无主语省略歧义"},
    {"key": "structure",      "name": "简历结构",   "weight": 0.15,
     "focus": "模块完整性、排版逻辑、信息密度、重要内容是否放前面"},
    {"key": "quantification", "name": "量化程度",   "weight": 0.10,
     "focus": "性能指标、用户量、优化幅度等量化数据的使用情况"},
    {"key": "authenticity",   "name": "真实可信度", "weight": 0.05,
     "focus": "表述是否夸大、技术深度描述是否与经验年限匹配、时间线是否合理"},
]


# ════════════════════════════════════════════════════════════════
# 节点①：upload_to_minio — 上传 PDF 到 MinIO 对象存储
# ════════════════════════════════════════════════════════════════
async def upload_to_minio_node(state: ResumeState) -> dict:
    """把本地临时 PDF 上传到 MinIO。MinIO 不可用时降级跳过。"""
    from backend.core.minio_client import upload_to_minio
    try:
        local_path = state["pdf_local_path"]
        remote_key = f"resumes/{state['student_id']}/{state['review_id']}.pdf"
        minio_path = await upload_to_minio(local_path, remote_key)
        logger.info("upload_to_minio.done", review_id=state["review_id"], minio_path=minio_path)
        return {"pdf_minio_path": minio_path}
    except Exception as e:
        logger.warning("upload_to_minio.unavailable", error=str(e))
        return {}


async def download_pdf_node(state: ResumeState) -> dict:
    """从 MinIO 下载 PDF 到本地。MinIO 不可用时降级跳过。"""
    import tempfile
    from backend.core.minio_client import download_from_minio
    try:
        minio_path = state["pdf_minio_path"]
        if not minio_path or "minio://" not in str(minio_path):
            return {}
        remote_key = str(minio_path).replace("minio://resumes/", "resumes/")
        tmp_path = os.path.join(tempfile.gettempdir(), f"{state['review_id']}_download.pdf")
        local_path = await download_from_minio(remote_key, tmp_path)
        logger.info("download_pdf.done", review_id=state["review_id"], local_path=local_path)
        return {"pdf_local_path": local_path}
    except Exception as e:
        logger.warning("download_pdf.unavailable", error=str(e))
        return {}


# ════════════════════════════════════════════════════════════════
# 节点③：extract_text —— PDF → 纯文本（双栏布局 + 线程池）
# ════════════════════════════════════════════════════════════════
def _sync_extract_text(pdf_path: str) -> dict:
    """同步 PDF 文本提取（在线程池中运行）。自动处理单栏/双栏布局。"""
    import fitz                         # PyMuPDF，导入名是 fitz
    doc = fitz.open(pdf_path)           # 打开 PDF 文件
    page_count = len(doc)               # 总页数
    all_text_parts = []                 # 每页文本收集到这里

    for page in doc:                    # 逐页处理
        blocks = page.get_text("blocks")              # 每块 = (x0,y0,x1,y1,text,block_no,type)
        text_blocks = [b for b in blocks if b[6] == 0]  # type==0 → 文字块（排除图片块）
        if not text_blocks:                           # 纯图片页 → 跳过
            continue

        page_width = page.rect.width                  # 页面宽度（用于判断双栏）
        midpoint   = page_width / 2                   # 水平中线
        left_blocks  = [b for b in text_blocks if b[0] < midpoint - 20]   # 偏左 20px 容差
        right_blocks = [b for b in text_blocks if b[0] >= midpoint - 20]  # 偏右
        is_two_column = (                             # 双栏判定：左右块数 ≥ 2 且右侧占比 > 30%
            len(left_blocks) >= 2 and len(right_blocks) >= 2
            and len(right_blocks) / max(len(text_blocks), 1) > 0.3
        )

        if is_two_column:                             # 双栏：左栏读完读右栏，各自按 y 坐标排序
            left_sorted  = sorted(left_blocks,  key=lambda b: b[1])
            right_sorted = sorted(right_blocks, key=lambda b: b[1])
            page_text = (
                "\n".join(b[4].strip() for b in left_sorted  if b[4].strip())
                + "\n"
                + "\n".join(b[4].strip() for b in right_sorted if b[4].strip())
            )
        else:                                         # 单栏：按 y 坐标从上到下
            sorted_blocks = sorted(text_blocks, key=lambda b: b[1])
            page_text = "\n".join(b[4].strip() for b in sorted_blocks if b[4].strip())

        all_text_parts.append(page_text)

    doc.close()
    raw_text = "\n\n---PAGE BREAK---\n\n".join(all_text_parts)  # 多页间插入分页标记，便于后续区分
    return {"raw_text": raw_text, "page_count": page_count}


async def extract_text_node(state: ResumeState) -> dict:
    """异步节点：用线程池跑同步解析，避免阻塞事件循环。"""
    pdf_path = state["pdf_local_path"]                # PDF 本地路径（上传接口已写入）
    try:
        loop = asyncio.get_running_loop()             # 拿到当前事件循环
        result = await loop.run_in_executor(None, _sync_extract_text, pdf_path)  # 丢到线程池执行
        raw_text, page_count = result["raw_text"], result["page_count"]

        if len(raw_text.strip()) < 200:               # 文字过少 → 可能是扫描件/图片 PDF，仅告警
            logger.warning("extract_text.text_too_short", text_length=len(raw_text.strip()))

        logger.info("extract_text.done", page_count=page_count, text_length=len(raw_text))
        return {"raw_text": raw_text, "page_count": page_count}
    except Exception as e:
        logger.error("extract_text.failed", error=str(e))
        raise


# ════════════════════════════════════════════════════════════════
# 节点④：extract_structured —— LLM 结构化提取
# 把 4000 字原文 → 结构化 dict（姓名/技能/项目/教育/工作）
# ════════════════════════════════════════════════════════════════
async def extract_structured_node(state: ResumeState) -> dict:
    """用 LLM Function Calling 把原始文本提取成 ResumeStructured 结构。"""
    raw_text = state["raw_text"]
    text_for_llm = raw_text[:4000] if len(raw_text) > 4000 else raw_text   # 超长截断，防止超出 token 上限
    prompt = EXTRACT_STRUCTURED_PROMPT.format(resume_text=text_for_llm)    # 填占位符 {resume_text}
    structured_llm = get_structured_llm("resume", ResumeStructured)        # 绑死 ResumeStructured 输出结构

    structured_dict = None
    for attempt in range(2):                           # 结构化输出偶发返回 None → 判空 + 重试
        try:
            result = await structured_llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),   # 系统角色
                HumanMessage(content=prompt),           # 简历原文 + 提取要求
            ])
            if result is None:                         # LLM 没调 tool → 返回 None
                raise ValueError("structured output returned None")
            structured_dict = result.model_dump()      # Pydantic 对象 → Python dict
            break
        except Exception as e:
            if attempt == 0:
                logger.warning("extract_structured.retry", error=str(e))
                await asyncio.sleep(1)
            else:
                logger.warning("extract_structured.failed", error=str(e))

    if structured_dict is None:                        # 两次都失败 → 降级为一个几乎空的结构
        structured_dict = ResumeStructured(name="未能提取").model_dump()

    logger.info("extract_structured.done",
                name=structured_dict.get("name", ""),
                projects_count=len(structured_dict.get("projects", [])))
    return {"structured": structured_dict}


# ════════════════════════════════════════════════════════════════
# 节点⑤：run_six_dimensions —— 六维度并行评审
# 6 个维度同时发 LLM 请求 → gather 等结果 → 加权算总分
# ════════════════════════════════════════════════════════════════
async def run_six_dimensions_node(state: ResumeState) -> dict:
    """六维度并行评审：asyncio.gather 同时评 6 个维度，算加权综合分。"""
    raw_text   = state["raw_text"]                     # PDF 原始文本
    structured = state.get("structured") or {}         # 结构化简历（可能为空）
    structured_summary = _build_structured_summary(structured)  # 压成 3-5 行摘要，塞 prompt 省 token

    async def review_one_dimension(dim: dict) -> dict:
        """评审单个维度，最多 2 次尝试，异常则降级 50 分。"""
        prompt_template = DIMENSION_REVIEW_PROMPTS.get(dim["key"], "")  # 按 key 取提示词模板
        if not prompt_template:                                        # key 不存在 → 直接降级
            return _empty_dimension_score(dim)

        prompt = prompt_template.format(                # 填占位符，拼出完整 prompt
            resume_text=raw_text[:3000],                # 截前 3000 字
            structured_summary=structured_summary,      # 简历摘要
            focus=dim["focus"],                         # 当前维度的评审侧重点
        )

        for attempt in range(2):                        # 最多 2 次尝试
            try:
                structured_llm = get_structured_llm("resume", DimensionScore)  # 绑死 DimensionScore
                result: DimensionScore = await structured_llm.ainvoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ])
                d = result.model_dump()                 # Pydantic 对象 → dict
                d["dimension"], d["weight"], d["key"] = dim["name"], dim["weight"], dim["key"]  # 代码层覆盖
                return d
            except Exception as e:
                if attempt == 0:                        # 第一次失败 → 等 1 秒再试
                    logger.warning("six_dimensions.dimension_retry", dimension=dim["name"], error=str(e))
                    await asyncio.sleep(1)
                else:                                   # 第二次仍失败 → 降级 50 分
                    logger.warning("six_dimensions.dimension_failed", dimension=dim["name"], error=str(e))
                    return _empty_dimension_score(dim)

    # 六维度并行：建 6 个协程 → gather 同时发 → 等全部返回
    tasks = [review_one_dimension(dim) for dim in SIX_DIMENSIONS]
    dimension_scores = await asyncio.gather(*tasks)

    # 加权综合分 = Σ(维度得分 × 维度权重)，权重和 = 1.0
    weighted_score = sum(d["score"] * d["weight"] for d in dimension_scores)

    logger.info("six_dimensions.done", weighted_score=round(weighted_score, 2),
                scores={d["key"]: d["score"] for d in dimension_scores})
    return {"dimension_scores": list(dimension_scores), "weighted_score": round(weighted_score, 2)}


# ════════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════════
def _build_structured_summary(structured: dict) -> str:
    """把结构化简历 dict 压成 5 行以内字符串。塞 prompt 里替代原文，省 token。"""
    lines = []
    if structured.get("name"):                          # 姓名
        lines.append(f"姓名：{structured['name']}")
    if structured.get("target_position"):               # 求职意向
        lines.append(f"求职意向：{structured['target_position']}")
    if structured.get("education"):                     # 最高学历（取第一段）
        edu = structured["education"][0]
        lines.append(f"最高学历：{edu.get('school','')} {edu.get('major','')} {edu.get('degree','')}")
    if structured.get("skills_list"):                   # 技术栈（最多列 10 个）
        lines.append(f"技术栈：{', '.join(structured['skills_list'][:10])}")
    if structured.get("projects"):                      # 项目经历（数量 + 前 3 个项目名）
        proj_names = [p.get("name", "") for p in structured["projects"]]
        lines.append(f"项目数量：{len(structured['projects'])} 个（{', '.join(proj_names[:3])}）")
    if structured.get("work_experience"):               # 工作经历段数
        lines.append(f"工作经历：{len(structured['work_experience'])} 段")
    return "\n".join(lines) if lines else "（结构化提取失败，请基于原文评审）"


def _empty_dimension_score(dim: dict) -> dict:
    """维度评审失败时的降级兜底：中性 50 分 + 人工复核标记。"""
    return {"key": dim["key"], "dimension": dim["name"], "score": 50, "weight": dim["weight"],
            "issues": ["该维度评审失败，建议人工复核"], "suggestions": []}


# ════════════════════════════════════════════════════════════════
# 节点⑥：diagnose_issues —— 逐条问题诊断
# Think 前置推理 → 结构化生成 IssueList → 按优先级排序
# ════════════════════════════════════════════════════════════════
async def diagnose_issues_node(state: ResumeState) -> dict:
    """汇总维度问题 → Think 前置推理 → 结构化生成问题清单 → 按优先级排序。"""
    dimension_scores = state.get("dimension_scores", [])  # 六维度评分结果（节点⑤产出）
    raw_text         = state["raw_text"]                   # 简历原文
    structured       = state.get("structured") or {}       # 结构化简历

    # ── ① 汇总各维度已发现的问题 ──
    all_raw_issues = []
    for dim in dimension_scores:
        for issue_text in dim.get("issues", []):          # 遍历每个维度下的问题列表
            all_raw_issues.append(f"[{dim['dimension']}] {issue_text}")  # 带维度前缀
    raw_issues_text = "\n".join(f"- {i}" for i in all_raw_issues) or "（暂无）"

    # ── ② Think 前置推理（自由文本宏观分析；失败不阻塞主流程）──
    reasoning_trace = ""
    try:
        dimension_scores_summary = "\n".join(              # 六维度评分摘要
            f"- {d['dimension']}：{d['score']}分 — 问题：{', '.join(d.get('issues', [])[:2])}"
            for d in dimension_scores
        )
        think_prompt = DIAGNOSE_THINK_PROMPT.format(       # 填 Think 提示词
            dimension_scores_summary=dimension_scores_summary, raw_issues=raw_issues_text)
        think_llm = get_llm("resume", temperature=0)       # 普通模型，不绑结构，自由输出分析
        think_resp = await think_llm.ainvoke([HumanMessage(content=think_prompt)])
        reasoning_trace = (                                # 兼容不同版本：优先 .text，否则 .content
            think_resp.text if hasattr(think_resp, "text") and not callable(think_resp.text)
            else str(think_resp.content)
        ).strip()
    except Exception as e:
        logger.warning("diagnose_think.failed", error=str(e))

    think_context = f"\n\n【诊断前宏观分析】\n{reasoning_trace}" if reasoning_trace else ""

    # ── ③ 结构化生成问题清单（IssueList）──
    prompt = DIAGNOSE_ISSUES_PROMPT.format(
        resume_text=raw_text[:3000],
        structured_summary=_build_structured_summary(structured),
        raw_issues=raw_issues_text,
    ) + think_context                                    # 拼上 Think 分析结果作为附加上下文

    try:
        structured_llm = get_structured_llm("resume", IssueList)  # 绑死 IssueList 结构
        result: IssueList = await structured_llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
        issues = [item.model_dump() for item in result.items]     # 每条 IssueItem → dict
    except Exception as e:
        logger.warning("diagnose_issues.failed", error=str(e))
        # 降级：直接用各维度原始问题，全部标 medium 优先级
        issues = [
            {"priority": "medium", "dimension": dim["dimension"], "description": issue,
             "location": "简历全文", "suggestion": "请参考评审建议修改"}
            for dim in dimension_scores for issue in dim.get("issues", [])
        ]

    # ── ④ 按优先级排序：high(0) → medium(1) → low(2) ──
    priority_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))

    logger.info("diagnose_issues.done", total=len(issues),
                high=sum(1 for i in issues if i.get("priority") == "high"))
    return {"issues": issues}


# ════════════════════════════════════════════════════════════════
# 节点⑦：generate_summary —— 整体评价
# 综合结构化信息 + 六维度评分 + 高优先级问题 → 生成评价报告
# ════════════════════════════════════════════════════════════════
async def generate_summary_node(state: ResumeState) -> dict:
    """综合结构化信息、评分、问题，生成面向学员的整体评价。"""
    structured       = state.get("structured") or {}      # 结构化简历
    dimension_scores = state.get("dimension_scores", [])  # 六维度评分
    issues           = state.get("issues", [])            # 问题清单（节点⑥产出）
    weighted_score   = state.get("weighted_score", 0.0)   # 加权综合分

    # ── ① 挑出高优先级问题，最多 5 条 ──
    high_issues = [i["description"] for i in issues if i.get("priority") == "high"][:5]
    high_issues_text = "\n".join(f"- {i}" for i in high_issues) or "（无高优先级问题）"

    # ── ② 六维度得分整理成可读文字 ──
    scores_text = "\n".join(
        f"- {d['dimension']}：{d['score']}分（权重{int(d['weight'] * 100)}%）" for d in dimension_scores)

    # ── ③ 拼提示词 ──
    prompt = GENERATE_SUMMARY_PROMPT.format(
        structured_summary=_build_structured_summary(structured),
        scores_summary=scores_text, weighted_score=round(weighted_score, 1),
        high_issues=high_issues_text, target_position=structured.get("target_position", "后端开发"),
    )

    structured_llm = get_structured_llm("resume", ResumeSummary)   # 绑死 ResumeSummary 结构

    summary_dict = None
    for attempt in range(2):                           # 结构化输出偶发返回 None → 判空 + 重试
        try:
            result = await structured_llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
            if result is None:                         # LLM 没调 tool → 返回 None
                raise ValueError("structured output returned None")
            summary_dict = result.model_dump()
            break
        except Exception as e:
            if attempt == 0:
                logger.warning("generate_summary.retry", error=str(e))
                await asyncio.sleep(1)
            else:
                logger.warning("generate_summary.failed", error=str(e))

    if summary_dict is None:                           # 两次都失败 → 兜底默认评价
        summary_dict = {
            "highlights": ["简历内容已完整提交"],
            "core_improvements": high_issues[:2] if high_issues else ["请参考各维度建议修改"],
            "overall_comment": f"综合评分 {round(weighted_score, 1)} 分，请参考各维度详细反馈。",
            "fit_assessment": "与目标岗位匹配度评估暂不可用",
        }

    logger.info("generate_summary.done", highlights_count=len(summary_dict.get("highlights", [])))
    return {"summary": summary_dict}


# ════════════════════════════════════════════════════════════════
# 节点⑧：save_results —— 持久化结果到数据库 + 清理临时文件
# ════════════════════════════════════════════════════════════════
async def save_results_node(state: ResumeState) -> dict:
    """把完整结果写入 resume_reviews 表（JSONB 列），清理本地临时 PDF。"""
    review_id = state["review_id"]                     # 本次审查的唯一 ID

    # 留一份完整结果给上层（API 可直接返回）
    structured_output = {
        "review_id": review_id, "student_id": state["student_id"],
        "structured": state.get("structured"),         # 结构化简历
        "weighted_score": state.get("weighted_score", 0),  # 加权综合分
        "dimension_scores": state.get("dimension_scores", []),  # 六维度详情
        "issues": state.get("issues", []),             # 问题清单
        "summary": state.get("summary"),               # 整体评价
    }

    async with AsyncSessionLocal() as session:          # 拿一个异步数据库会话
        try:
            await session.execute(                      # 执行 UPDATE，写 JSONB 字段
                text("""
                    UPDATE resume_reviews
                    SET structured_data = :structured_data,
                        scores          = :scores,
                        issues          = :issues,
                        summary         = :summary,
                        status          = 'done',
                        updated_at      = NOW()
                    WHERE id = :review_id
                """),
                {
                    # JSONB 列：先 json.dumps 转 JSON 字符串；ensure_ascii=False 保留中文原文
                    "structured_data": json.dumps(state.get("structured"), ensure_ascii=False),
                    "scores": json.dumps(
                        {"dimension_scores": state.get("dimension_scores", []),
                         "weighted_score": state.get("weighted_score", 0)},
                        ensure_ascii=False),
                    "issues":  json.dumps(state.get("issues", []), ensure_ascii=False),
                    "summary": json.dumps(state.get("summary"), ensure_ascii=False),
                    "review_id": review_id,
                },
            )
            await session.commit()                     # 提交事务
            logger.info("save_results.db_written", review_id=review_id)
        except Exception as e:
            await session.rollback()                   # 异常回滚
            logger.error("save_results.db_failed", error=str(e))
            raise

    # 清理本地临时 PDF（上传接口生成的临时文件）
    local_path = state.get("pdf_local_path", "")
    if local_path and os.path.exists(local_path):
        os.remove(local_path)
        logger.info("save_results.tmp_cleaned", path=local_path)

    return {"fallback_used": False, "structured_output": structured_output}


# ════════════════════════════════════════════════════════════════
# 模块自测：实测 PDF 文本提取（离线，不依赖 LLM / 数据库）
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import fitz, tempfile                          # PyMuPDF + 系统临时目录
    _pdf_path = os.path.join(tempfile.gettempdir(), "sample_resume.pdf")
    _doc = fitz.open(); _page = _doc.new_page()    # 创建空白 PDF
    _page.insert_text((50, 60),                     # 写入假简历文本
        "张三  后端开发\n技能\n熟悉 Java、Spring Boot、Redis\n"
        "项目经历\n电商系统  2023.06 - 2023.12\nQPS 提升 30%",
        fontsize=11, fontname="china-s")           # 中文字体
    _doc.save(_pdf_path); _doc.close()             # 保存到系统临时目录（自动适配 Win/Linux）
    _r = asyncio.run(extract_text_node({"pdf_local_path": _pdf_path}))
    print("page_count:", _r["page_count"], "| 含 Spring Boot:", "Spring Boot" in _r["raw_text"])
