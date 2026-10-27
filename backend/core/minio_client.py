"""
minio_client - MinIO 对象存储客户端

"""
import os
from minio import Minio
from minio.error import S3Error

from backend.core.logger import get_logger

logger = get_logger(__name__)

BUCKET_NAME = "resumes"

_endpoint  = os.getenv("MINIO_ENDPOINT",  "localhost:9000")
_access    = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
_secret    = os.getenv("MINIO_SECRET_KEY", "minioadmin")
_secure    = os.getenv("MINIO_SECURE",     "false").lower() == "true"

_client = Minio(_endpoint, access_key=_access, secret_key=_secret, secure=_secure)


def _ensure_bucket():
    try:
        if not _client.bucket_exists(BUCKET_NAME):
            _client.make_bucket(BUCKET_NAME)
            logger.info("minio.bucket_created", bucket=BUCKET_NAME)
        else:
            logger.info("minio.bucket_ready", bucket=BUCKET_NAME)
    except S3Error as e:
        logger.warning("minio.bucket_error", error=str(e))


async def upload_to_minio(local_path: str, remote_key: str) -> str:
    import asyncio
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _ensure_bucket)

    def _upload():
        _client.fput_object(BUCKET_NAME, remote_key, local_path, content_type="application/pdf")

    await loop.run_in_executor(None, _upload)
    path = f"minio://{BUCKET_NAME}/{remote_key}"
    logger.info("minio.upload_done", key=remote_key)
    return path


async def download_from_minio(remote_key: str, local_path: str) -> str:
    import asyncio
    loop = asyncio.get_running_loop()

    def _download():
        _client.fget_object(BUCKET_NAME, remote_key, local_path)

    await loop.run_in_executor(None, _download)
    logger.info("minio.download_done", key=remote_key)
    return local_path
