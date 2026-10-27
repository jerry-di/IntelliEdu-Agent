"""
config - 配置

"""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """配置"""
    db_host: str = 'localhost'
    db_port: int = 5433
    db_name: str = 'eduagent'
    db_user: str
    db_password: str

    @property
    def database_url(self) -> str:
        """获取数据库 URL"""
        return f'postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}'

    milvus_host: str = 'localhost'
    milvus_port: int = 19531

    deepseek_api_key: str
    deepseek_base_url: str = 'https://api.deepseek.com'
    deepseek_model_chat: str = 'deepseek-v4-flash'
    deepseek_model_coder: str = 'deepseek-v4-flash'

    reranker_model_path: str = './models/reranker/bge-reranker-large'
    classifier_model_path: str = './models/classifier/all-MiniLM-L6-v2'
    finetuned_classifier_path: str = './models/classifier/query-classifier-finetuned'
    bge_m3_model_path: str = './models/embedding/bge-m3'

    jwt_secret_key: str
    jwt_algorithm: str = 'HS256'
    jwt_access_token_expire_minutes: int = 10080

    kb_mcp_server_url:  str = 'http://localhost:8000/mcp/kb'
    web_search_mcp_url: str = 'http://localhost:8000/mcp/web-search'

    tavily_api_key: str = ''

    moonshot_api_key: str = ''
    moonshot_base_url: str = 'https://api.moonshot.cn/v1'
    anthropic_auth_token: str = ''
    anthropic_base_url: str = 'https://api.cubence.com'

    app_env: str = 'local'
    app_debug: bool = False
    app_host: str = '127.0.0.1'
    app_port: int = 8000
    log_level: str = 'INFO'
    default_tenant_id: str = 'tenant_default'

    class Config:
        """Pydantic 的元配置"""
        # 基于 config.py 所在目录定位 .env.local，而非当前工作目录
        env_file = os.path.join(os.path.dirname(__file__), '..', '.env.local')
        env_file_encoding = 'utf-8'
        case_sensitive = False
        extra = 'ignore'


@lru_cache()
def get_settings() -> Settings:
    """获取全局唯一的配置对象"""
    return Settings()