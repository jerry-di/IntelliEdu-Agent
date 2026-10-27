"""
migrations - 自动迁移补丁

"""

# backend/db/migrations.py
from sqlalchemy import text
from backend.dependencies import AsyncSessionLocal
from backend.core.logger import get_logger

logger = get_logger(__name__)

# 所有补丁，按时间顺序追加。SQL 必须幂等（带 IF NOT EXISTS）
_MIGRATIONS: list[tuple[str, str]] = [
    (
        "exam_submissions.weak_points",
        "ALTER TABLE exam_submissions ADD COLUMN IF NOT EXISTS weak_points JSONB",
    ),
    (
        "exam_reviews.knowledge_tag",
        "ALTER TABLE exam_reviews ADD COLUMN IF NOT EXISTS knowledge_tag VARCHAR(128)",
    ),
    (
        "idx_exam_submissions_student_created",
        "CREATE INDEX IF NOT EXISTS idx_exam_submissions_student_created "
        "ON exam_submissions (student_id, created_at DESC)",
    ),
    (
        "resume_reviews.table",
        """
        CREATE TABLE IF NOT EXISTS resume_reviews (
            id              UUID PRIMARY KEY,
            tenant_id       VARCHAR(64)   NOT NULL,
            student_id      VARCHAR(64)   NOT NULL,
            pdf_minio_path  VARCHAR(512),
            status          VARCHAR(16)   NOT NULL DEFAULT 'processing',
            structured_data JSONB,
            scores          JSONB,
            issues          JSONB,
            summary         JSONB,
            error_msg       VARCHAR(1000),
            created_at      TIMESTAMP     NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMP     NOT NULL DEFAULT NOW()
        )
        """,
    ),
    (
        "idx_resume_reviews_student",
        "CREATE INDEX IF NOT EXISTS idx_resume_reviews_student "
        "ON resume_reviews (student_id, created_at DESC)",
    ),
    (
        "qa_sessions.table",
        """
        CREATE TABLE IF NOT EXISTS qa_sessions (
            id              UUID PRIMARY KEY,
            tenant_id       VARCHAR(64)  NOT NULL,
            student_id      VARCHAR(64)  NOT NULL,
            thread_id       VARCHAR(128) NOT NULL UNIQUE,
            summary         TEXT,
            summary_version INTEGER      DEFAULT 1,
            created_at      TIMESTAMP    NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMP    NOT NULL DEFAULT NOW()
        )
        """,
    ),
    (
        "knowledge_pending_queue.table",
        """
        CREATE TABLE IF NOT EXISTS knowledge_pending_queue (
            id          UUID PRIMARY KEY,
            tenant_id   VARCHAR(64)  NOT NULL,
            question    TEXT         NOT NULL,
            student_id  VARCHAR(64)  NOT NULL,
            confidence  FLOAT,
            status      VARCHAR(16)  NOT NULL DEFAULT 'pending',
            created_at  TIMESTAMP    NOT NULL DEFAULT NOW()
        )
        """,
    ),
    (
        "interview_sessions.table",
        """
        CREATE TABLE IF NOT EXISTS interview_sessions (
            id              UUID PRIMARY KEY,
            tenant_id       VARCHAR(64)  NOT NULL,
            student_id      VARCHAR(64)  NOT NULL,
            session_id      VARCHAR(64)  NOT NULL,
            thread_id       VARCHAR(128) NOT NULL UNIQUE,
            target_position VARCHAR(128) NOT NULL,
            resume_review_id UUID,
            summary         TEXT,
            report          JSONB,
            overall_score   INTEGER,
            status          VARCHAR(16)  NOT NULL DEFAULT 'in_progress',
            finished_at     TIMESTAMP,
            created_at      TIMESTAMP    NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMP    NOT NULL DEFAULT NOW()
        )
        """,
    ),
]


async def run_migrations() -> None:
    """应用启动时执行所有 Schema 补丁；单条失败只警告，不阻断启动。"""
    async with AsyncSessionLocal() as session:
        for desc, sql in _MIGRATIONS:
            try:
                await session.execute(text(sql))
                await session.commit()
            except Exception as e:
                await session.rollback()
                if "already exists" not in str(e):
                    logger.warning("db.migration_failed", column=desc, error=str(e))
    logger.info("db.migrations_done", count=len(_MIGRATIONS))
