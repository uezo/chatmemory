import asyncio
import os
import uuid
import pytest
from chatmemory import ChatMemory, HistoryMessage


DB_NAME = os.getenv("DB_NAME", "your_db")
DB_USER = os.getenv("DB_USER", "your_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 5432)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Fixture to instantiate ChatMemory with real configuration.
# Be sure to configure these parameters appropriately for your environment.
@pytest.fixture(scope="module")
def chat_memory():
    cm = ChatMemory(
        openai_api_key=OPENAI_API_KEY,
        openai_base_url=OPENAI_BASE_URL,
        llm_model=LLM_MODEL,
        db_name=DB_NAME,
        db_user=DB_USER,
        db_password=DB_PASSWORD,
        db_port=DB_PORT
    )
    yield cm

def test_add_and_get_history(chat_memory):
    # Generate unique user_id and session_id for isolation.
    user_id = str(uuid.uuid4())
    session_id = f"session_{uuid.uuid4()}"
    messages = [
        HistoryMessage(role="user", content="Hello", metadata={}),
        HistoryMessage(role="assistant", content="Hi there!", metadata={}),
    ]
    # Add history
    chat_memory.add_history(user_id, session_id, messages)
    # Retrieve history and verify
    history = chat_memory.get_history(user_id=user_id, session_id=session_id)
    assert len(history) == len(messages)
    # Cleanup: delete history
    chat_memory.delete_history(user_id=user_id, session_id=session_id)
    history_after = chat_memory.get_history(user_id=user_id, session_id=session_id)
    assert len(history_after) == 0

def test_create_summary(chat_memory):
    user_id = str(uuid.uuid4())
    session_id = f"session_{uuid.uuid4()}"
    messages = [
        HistoryMessage(role="user", content="How is the weather today?", metadata={}),
        HistoryMessage(role="assistant", content="It is sunny.", metadata={})
    ]
    chat_memory.add_history(user_id, session_id, messages)
    # Create summary (this calls the LLM and embeddings)
    asyncio.run(chat_memory.create_summary(user_id, session_id))
    # Retrieve summaries and verify at least one exists
    summaries = chat_memory.get_summaries(user_id=user_id, session_id=session_id)
    assert len(summaries) >= 1
    # Cleanup: delete history and summaries
    chat_memory.delete_history(user_id=user_id, session_id=session_id)
    chat_memory.delete_summaries(user_id=user_id, session_id=session_id)

def test_add_get_delete_knowledge(chat_memory):
    user_id = str(uuid.uuid4())
    knowledge_text = "Test knowledge content for unit testing."
    # Add knowledge record
    asyncio.run(chat_memory.add_knowledge(user_id, knowledge_text))
    knowledges = chat_memory.get_knowledge(user_id)
    assert any(knowledge_text in k.knowledge for k in knowledges)
    # Delete knowledge for this user
    chat_memory.delete_knowledge(user_id)
    knowledges_after = chat_memory.get_knowledge(user_id)
    assert len(knowledges_after) == 0

def test_search(chat_memory):
    user_id = str(uuid.uuid4())
    session_id = f"session_{uuid.uuid4()}"
    # Add conversation history that should be searchable.
    messages = [
        HistoryMessage(role="user", content="Tell me a joke", metadata={}),
        HistoryMessage(role="assistant", content="Why did the chicken cross the road?", metadata={})
    ]
    chat_memory.add_history(user_id, session_id, messages)
    # Create summary so that search has something to work with.
    asyncio.run(chat_memory.create_summary(user_id, session_id))
    # Perform search.
    search_result = asyncio.run(
        chat_memory.search(user_id, "joke", top_k=3, search_content=False, include_retrieved_data=True)
    )
    # Verify that a non-empty answer was returned.
    assert search_result is not None
    assert isinstance(search_result.answer, str)
    # Cleanup
    chat_memory.delete_history(user_id=user_id, session_id=session_id)
    chat_memory.delete_summaries(user_id=user_id, session_id=session_id)
