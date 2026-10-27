import asyncio
import os
import uuid

import asyncpg
from dotenv import load_dotenv
from passlib.context import CryptContext

load_dotenv('.env.local')

DB_DSN = f'postgresql://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}'
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


async def seed_users():
    """灌入 4 个测试账号"""
    conn = await asyncpg.connect(DB_DSN)
    print('[OK] 数据库连接成功，开始灌入测试账号')
    try:
        users = [
            {'username': 'admin', 'email': 'admin@eduagent.local', 'pwd': 'Admin@123456', 'role': 'admin'},
            {'username': 'teacher01', 'email': 'teacher01@eduagent.local', 'pwd': 'Teacher@123456', 'role': 'teacher'},
            {'username': 'student01', 'email': 'student01@eduagent.local', 'pwd': 'Student@123456', 'role': 'student'},
            {'username': 'student02', 'email': 'student02@eduagent.local', 'pwd': 'Student@123456', 'role': 'student'},
        ]
        async with conn.transaction():
            for u in users:
                await conn.execute(
                    '''
                    INSERT INTO users
                        (id, tenant_id, username, email, password_hash, role)
                    VALUES
                        ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (tenant_id, email) DO NOTHING
                    ''',
                    str(uuid.uuid4()),
                    'tenant_default',
                    u['username'],
                    u['email'],
                    pwd_context.hash(u['pwd']),
                    u['role'],
                )
            print('[OK] 测试账号灌入完成！！！')
    finally:
        await conn.close()


if __name__ == '__main__':
    asyncio.run(seed_users())
