import os
from fastapi import FastAPI
from chatmemory import ChatMemory

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

cm = ChatMemory(
    openai_api_key=OPENAI_API_KEY,
    openai_base_url=OPENAI_BASE_URL,
    llm_model=LLM_MODEL,
    embedding_model=EMBEDDING_MODEL,
    db_name=DB_NAME,
    db_user=DB_USER,
    db_password=DB_PASSWORD,
    db_host="chatmemory-db",
    db_port=5432,
)

app = FastAPI(title="ChatMemory", version="0.2.1")
app.include_router(cm.get_router())
