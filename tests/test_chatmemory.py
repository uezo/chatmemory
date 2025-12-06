import asyncio
import datetime
import os
import uuid
import pytest
from chatmemory.chatmemory import ChatMemory, HistoryMessage, Diary

from dotenv import load_dotenv
load_dotenv()

DB_NAME = os.getenv("DB_NAME", "your_db")
DB_USER = os.getenv("DB_USER", "your_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 5432)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "3072"))

# Fixture to instantiate ChatMemory with real configuration.
# Be sure to configure these parameters appropriately for your environment.
@pytest.fixture(scope="module")
def chat_memory():
    cm = ChatMemory(
        openai_api_key=OPENAI_API_KEY,
        openai_base_url=OPENAI_BASE_URL,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIMENSION,
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
    # Test within_seconds: add a new message, sleep, then check filtering
    import time
    msg_late = [HistoryMessage(role="user", content="Late message", metadata={})]
    chat_memory.add_history(user_id, session_id, msg_late)
    time.sleep(2)
    # Only the last message should be within 1 second (should be 0)
    history_recent = chat_memory.get_history(user_id=user_id, session_id=session_id, within_seconds=1)
    assert len(history_recent) == 0
    # All messages should be returned with within_seconds=0 (unlimited)
    history_all = chat_memory.get_history(user_id=user_id, session_id=session_id, within_seconds=0)
    assert len(history_all) == len(messages) + 1
    # Cleanup: delete history
    chat_memory.delete_history(user_id=user_id, session_id=session_id)
    history_after = chat_memory.get_history(user_id=user_id, session_id=session_id)
    assert len(history_after) == 0

def test_channel_field_functionality(chat_memory):
    """Test the channel field functionality for storing and retrieving messages."""
    # Generate unique user_id and session_id for isolation
    user_id = str(uuid.uuid4())
    session_id = f"session_{uuid.uuid4()}"
    
    # Create messages for different channels
    chatapp_messages = [
        HistoryMessage(role="user", content="Hello from ChatApp", metadata={}),
        HistoryMessage(role="assistant", content="Hi there from ChatApp!", metadata={}),
    ]
    
    discord_messages = [
        HistoryMessage(role="user", content="Hello from Discord", metadata={}),
        HistoryMessage(role="assistant", content="Hi there from Discord!", metadata={}),
    ]
    
    # Add history for both channels
    chat_memory.add_history(user_id, session_id, chatapp_messages, channel="chatapp")
    chat_memory.add_history(user_id, session_id, discord_messages, channel="discord")
    
    # Retrieve all history for the session
    all_history = chat_memory.get_history(user_id=user_id, session_id=session_id)
    assert len(all_history) == 4  # All messages from both channels
    
    # Retrieve history filtered by chatapp channel
    chatapp_history = chat_memory.get_history(user_id=user_id, session_id=session_id, channel="chatapp")
    assert len(chatapp_history) == 2
    assert all(msg.channel == "chatapp" for msg in chatapp_history)
    
    # Retrieve history filtered by discord channel
    discord_history = chat_memory.get_history(user_id=user_id, session_id=session_id, channel="discord")
    assert len(discord_history) == 2
    assert all(msg.channel == "discord" for msg in discord_history)
    
    # Delete only chatapp channel messages
    chat_memory.delete_history(user_id=user_id, session_id=session_id, channel="chatapp")
    
    # Verify only chatapp messages were deleted
    remaining_history = chat_memory.get_history(user_id=user_id, session_id=session_id)
    assert len(remaining_history) == 2  # Only discord messages should remain
    assert all(msg.channel == "discord" for msg in remaining_history)
    
    # Clean up: delete all remaining history
    chat_memory.delete_history(user_id=user_id, session_id=session_id)
    history_after = chat_memory.get_history(user_id=user_id, session_id=session_id)
    assert len(history_after) == 0

def test_get_sessions(chat_memory):
    """Test retrieving session details (SessionInfo) with the latest created_at and channel information."""
    user_id = str(uuid.uuid4())
    # Create multiple sessions with distinct session IDs.
    session_ids = [f"session_{uuid.uuid4()}" for _ in range(3)]

    # Add history for each session.
    for session_id in session_ids:
        messages = [
            HistoryMessage(role="user", content="Session test message", metadata={}),
            HistoryMessage(role="assistant", content="Test response", metadata={})
        ]
        chat_memory.add_history(user_id, session_id, messages)

    # Retrieve session details using the new method.
    sessions = chat_memory.get_sessions(user_id=user_id, within_seconds=3600, limit=5)

    # Verify that a session detail is returned for each created session.
    assert len(sessions) == len(session_ids)

    # Check that each SessionInfo has the correct user_id and a session_id from the created sessions.
    retrieved_session_ids = {info.session_id for info in sessions}
    assert set(session_ids) == retrieved_session_ids

    # Cleanup: Delete history for all created sessions.
    for session_id in session_ids:
        chat_memory.delete_history(user_id=user_id, session_id=session_id)

def test_get_sessions_with_filters(chat_memory):
    """Test retrieving session details (SessionInfo) verifying within_seconds and limit filters."""
    user_id = str(uuid.uuid4())
    # Create multiple sessions (7 sessions)
    session_ids = [f"session_{uuid.uuid4()}" for _ in range(7)]

    # Add history for each session.
    for session_id in session_ids:
        messages = [
            HistoryMessage(role="user", content="Session test message", metadata={}),
            HistoryMessage(role="assistant", content="Test response", metadata={})
        ]
        chat_memory.add_history(user_id, session_id, messages)

    # Update one session to have an old created_at (e.g., 2 hours ago) to test the within_seconds filter.
    old_session_id = session_ids[0]
    with chat_memory.get_db_cursor() as (cur, conn):
        cur.execute(
            "UPDATE conversation_history SET created_at = NOW() - INTERVAL '2 hours' WHERE session_id = %s",
            (old_session_id,)
        )
        conn.commit()

    # Verify within_seconds filter:
    # Using within_seconds=3600 (1 hour) should exclude the old session.
    sessions_within = chat_memory.get_sessions(user_id=user_id, within_seconds=3600, limit=10)
    retrieved_ids_within = {info.session_id for info in sessions_within}
    assert old_session_id not in retrieved_ids_within
    assert len(sessions_within) == len(session_ids) - 1  # Old session is excluded.

    # Verify limit filter:
    # Using a larger within_seconds to include all sessions but setting limit=5 should return only 5 sessions.
    sessions_limited = chat_memory.get_sessions(user_id=user_id, within_seconds=7200, limit=5)
    assert len(sessions_limited) == 5

    # Cleanup: Delete history for all created sessions.
    for session_id in session_ids:
        chat_memory.delete_history(user_id=user_id, session_id=session_id)

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
    # Retrieve summaries and verify just one exists
    summaries = chat_memory.get_summaries(user_id=user_id, session_id=session_id)
    assert len(summaries) == 1
    # Cleanup: delete history and summaries
    chat_memory.delete_history(user_id=user_id, session_id=session_id)
    chat_memory.delete_summaries(user_id=user_id, session_id=session_id)

def test_create_summaries(chat_memory):
    user_id = str(uuid.uuid4())
    session_id_1 = f"session_{uuid.uuid4()}"
    messages = [
        HistoryMessage(role="user", content="How is the weather today?", metadata={}),
        HistoryMessage(role="assistant", content="It is sunny.", metadata={})
    ]
    chat_memory.add_history(user_id, session_id_1, messages)
    session_id_2 = f"session_{uuid.uuid4()}"
    messages = [
        HistoryMessage(role="user", content="I'm very sleepy.", metadata={}),
        HistoryMessage(role="assistant", content="Go to the bed.", metadata={})
    ]
    chat_memory.add_history(user_id, session_id_2, messages)
    session_id_3 = f"session_{uuid.uuid4()}"
    messages = [
        HistoryMessage(role="user", content="Cat is better than dog.", metadata={}),
        HistoryMessage(role="assistant", content="Exactly.", metadata={})
    ]
    chat_memory.add_history(user_id, session_id_3, messages)

    # Create summary (this calls the LLM and embeddings)
    asyncio.run(chat_memory.create_summary(user_id))
    # Retrieve summaries and 2 exist
    summaries = chat_memory.get_summaries(user_id=user_id)
    assert len(summaries) == 2  # The latest one is skipped

    # Cleanup: delete history and summaries
    chat_memory.delete_history(user_id=user_id, session_id=session_id_1)
    chat_memory.delete_history(user_id=user_id, session_id=session_id_2)
    chat_memory.delete_history(user_id=user_id, session_id=session_id_3)
    chat_memory.delete_summaries(user_id=user_id, session_id=session_id_1)
    chat_memory.delete_summaries(user_id=user_id, session_id=session_id_2)
    chat_memory.delete_summaries(user_id=user_id, session_id=session_id_3)

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

def test_search_includes_diary(chat_memory, monkeypatch):
    """Search should retrieve diary embeddings alongside summaries/knowledge."""
    user_id = str(uuid.uuid4())
    fake_embedding = [0.01] * chat_memory.embedding_dimension

    async def fake_embed(text: str):
        return fake_embedding

    async def fake_llm(system_prompt: str, user_prompt: str):
        return "diary answer"

    monkeypatch.setattr(chat_memory, "embed", fake_embed)
    monkeypatch.setattr(chat_memory, "llm", fake_llm)

    diary_date = datetime.date.today()
    asyncio.run(
        chat_memory.add_diary(
            Diary(
                user_id=user_id,
                diary_date=diary_date,
                content="Today I saw shooting stars",
                metadata={"mood": "calm"},
            )
        )
    )

    search_result = asyncio.run(
        chat_memory.search(user_id, "stars", top_k=3, search_content=False, include_retrieved_data=True)
    )
    assert search_result is not None
    assert isinstance(search_result.answer, str)
    assert search_result.retrieved_data is not None
    assert "Diary" in search_result.retrieved_data
    assert "shooting stars" in search_result.retrieved_data

    chat_memory.delete_diary(user_id=user_id)

def test_diary_crud(chat_memory):
    """Diary add/get/update/delete, with since/until filtering on diary_date."""
    user_id = str(uuid.uuid4())
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    diary_today = Diary(user_id=user_id, diary_date=today, content="Today diary", metadata={"mood": "good"})
    diary_yesterday = Diary(user_id=user_id, diary_date=yesterday, content="Yesterday diary", metadata={"mood": "ok"})

    # Add two diaries
    asyncio.run(chat_memory.add_diary(diary_today))
    asyncio.run(chat_memory.add_diary(diary_yesterday))

    # Get by exact date
    diaries_exact = chat_memory.get_diaries(user_id=user_id, diary_date=today)
    assert len(diaries_exact) == 1
    assert diaries_exact[0].content == "Today diary"

    # since/until should use diary_date
    since_dt = datetime.datetime.combine(today, datetime.time.min)
    diaries_since = chat_memory.get_diaries(user_id=user_id, since=since_dt)
    assert all(d.diary_date >= today for d in diaries_since)

    until_dt = datetime.datetime.combine(yesterday, datetime.time.max)
    diaries_until = chat_memory.get_diaries(user_id=user_id, until=until_dt)
    assert all(d.diary_date <= yesterday for d in diaries_until)

    # Update today's diary content and metadata (embedding recomputed in impl)
    asyncio.run(
        chat_memory.update_diary(
            user_id=user_id,
            diary_date=today,
            content="Today diary updated",
            metadata={"mood": "great"},
        )
    )
    diaries_after_update = chat_memory.get_diaries(user_id=user_id, diary_date=today)
    assert diaries_after_update[0].content == "Today diary updated"
    assert diaries_after_update[0].metadata.get("mood") == "great"

    # Delete specific and all
    chat_memory.delete_diary(user_id=user_id, diary_date=today)
    remaining = chat_memory.get_diaries(user_id=user_id)
    assert all(d.diary_date != today for d in remaining)

    chat_memory.delete_diary(user_id=user_id)
    with pytest.raises(ValueError):
        chat_memory.get_diaries(user_id=None, diary_date=None)
