import datetime
import os
import uuid
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from chatmemory import ChatMemory

from dotenv import load_dotenv
load_dotenv()

# Create a ChatMemory instance with real configuration.
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

chat_memory = ChatMemory(
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

# Create a FastAPI app and include the ChatMemory router.
app = FastAPI()
app.include_router(chat_memory.get_router())

client = TestClient(app)

@pytest.fixture(scope="module")
def test_user():
    return str(uuid.uuid4())

@pytest.fixture(scope="module")
def test_session():
    return f"session_{uuid.uuid4()}"

def test_history_endpoints(test_user, test_session):
    # POST /history: Add conversation history
    payload = {
        "user_id": test_user,
        "session_id": test_session,
        "messages": [
            {"role": "user", "content": "Hello", "metadata": {}},
            {"role": "assistant", "content": "Hi there", "metadata": {}}
        ]
    }
    response = client.post("/history", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    # GET /history: Retrieve conversation history
    response = client.get("/history", params={"user_id": test_user, "session_id": test_session})
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data
    assert len(data["messages"]) == 2

    # GET /history/session_ids: Verify session id is returned
    response = client.get("/history/session_ids", params={"user_id": test_user})
    assert response.status_code == 200
    data = response.json()
    assert "session_ids" in data
    assert test_session in data["session_ids"]

def test_channel_field_api():
    """Test the channel field functionality through the API endpoints."""
    # Create unique user_id and session_id for this test to ensure isolation
    channel_test_user = str(uuid.uuid4())
    channel_test_session = f"session_{uuid.uuid4()}"
    
    # POST /history: Add conversation history with chatapp channel
    chatapp_payload = {
        "user_id": channel_test_user,
        "session_id": channel_test_session,
        "channel": "chatapp",
        "messages": [
            {"role": "user", "content": "Hello from ChatApp", "metadata": {}},
            {"role": "assistant", "content": "Hi there from ChatApp", "metadata": {}}
        ]
    }
    response = client.post("/history", json=chatapp_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    # POST /history: Add conversation history with discord channel
    discord_payload = {
        "user_id": channel_test_user,
        "session_id": channel_test_session,
        "channel": "discord",
        "messages": [
            {"role": "user", "content": "Hello from Discord", "metadata": {}},
            {"role": "assistant", "content": "Hi there from Discord", "metadata": {}}
        ]
    }
    response = client.post("/history", json=discord_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    # GET /history: Retrieve all conversation history
    response = client.get("/history", params={"user_id": channel_test_user, "session_id": channel_test_session})
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data
    assert len(data["messages"]) == 4  # All messages from both channels

    # GET /history: Retrieve conversation history filtered by chatapp channel
    response = client.get("/history", params={"user_id": channel_test_user, "session_id": channel_test_session, "channel": "chatapp"})
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data
    assert len(data["messages"]) == 2
    assert all(msg["channel"] == "chatapp" for msg in data["messages"])

    # GET /history: Retrieve conversation history filtered by discord channel
    response = client.get("/history", params={"user_id": channel_test_user, "session_id": channel_test_session, "channel": "discord"})
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data
    assert len(data["messages"]) == 2
    assert all(msg["channel"] == "discord" for msg in data["messages"])

    # DELETE /history: Delete only chatapp channel messages
    response = client.delete("/history", params={"user_id": channel_test_user, "session_id": channel_test_session, "channel": "chatapp"})
    assert response.status_code == 200
    assert response.json()["status"] == "history deleted"

    # GET /history: Verify only chatapp messages were deleted
    response = client.get("/history", params={"user_id": channel_test_user, "session_id": channel_test_session})
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data
    assert len(data["messages"]) == 2  # Only discord messages should remain
    assert all(msg["channel"] == "discord" for msg in data["messages"])

    # Clean up: delete all remaining history
    response = client.delete("/history", params={"user_id": channel_test_user, "session_id": channel_test_session})
    assert response.status_code == 200
    assert response.json()["status"] == "history deleted"

def test_get_sessions_endpoint():
    """Test the REST API endpoint for retrieving session details with time and limit filters."""
    # Create a unique user_id and multiple sessions.
    user_id = str(uuid.uuid4())
    session_ids = [f"session_{uuid.uuid4()}" for _ in range(7)]
    
    # Add history for each session via the API.
    for session_id in session_ids:
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "Session test message", "metadata": {}},
                {"role": "assistant", "content": "Test response", "metadata": {}}
            ]
        }
        response = client.post("/history", json=payload)
        assert response.status_code == 200
    
    # Update one session to have an old created_at (e.g., 2 hours ago)
    old_session_id = session_ids[0]
    with chat_memory.get_db_cursor() as (cur, conn):
        cur.execute(
            "UPDATE conversation_history SET created_at = NOW() - INTERVAL '2 hours' WHERE session_id = %s",
            (old_session_id,)
        )
        conn.commit()
    
    # Test within_seconds filter: using within_seconds=3600 (1 hour) should exclude the old session.
    params = {"user_id": user_id, "within_seconds": 3600, "limit": 10}
    response = client.get("/history/sessions", params=params)
    assert response.status_code == 200
    data = response.json()
    assert "sessions" in data
    sessions = data["sessions"]
    # The old session is filtered out.
    assert old_session_id not in {s["session_id"] for s in sessions}
    assert len(sessions) == len(session_ids) - 1
    
    # Test limit filter: using within_seconds=7200 (2 hours) and limit=5 should return only 5 sessions.
    params = {"user_id": user_id, "within_seconds": 7200, "limit": 5}
    response = client.get("/history/sessions", params=params)
    assert response.status_code == 200
    data = response.json()
    sessions = data["sessions"]
    assert len(sessions) == 5
    
    # Cleanup: Delete history for all created sessions.
    for session_id in session_ids:
        response = client.delete("/history", params={"user_id": user_id, "session_id": session_id})
        assert response.status_code == 200

def test_diary_endpoints():
    """Diary add/get/update/delete via REST API, including since/until filtering."""
    user_id = str(uuid.uuid4())
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    today_payload = {
        "user_id": user_id,
        "diary_date": today.isoformat(),
        "content": "Today diary",
        "metadata": {"mood": "good"},
    }
    yesterday_payload = {
        "user_id": user_id,
        "diary_date": yesterday.isoformat(),
        "content": "Yesterday diary",
        "metadata": {"mood": "ok"},
    }

    # Add two diaries
    response = client.post("/diary", json=today_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "diary upserted"

    response = client.post("/diary", json=yesterday_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "diary upserted"

    # Get by exact date
    response = client.get("/diary", params={"user_id": user_id, "diary_date": today.isoformat()})
    assert response.status_code == 200
    diaries_exact = response.json()["diaries"]
    assert len(diaries_exact) == 1
    assert diaries_exact[0]["content"] == "Today diary"

    # since/until should use diary_date
    since_dt = datetime.datetime.combine(today, datetime.time.min)
    response = client.get("/diary", params={"user_id": user_id, "since": since_dt.isoformat()})
    assert response.status_code == 200
    diaries_since = response.json()["diaries"]
    assert all(datetime.date.fromisoformat(d["diary_date"]) >= today for d in diaries_since)

    until_dt = datetime.datetime.combine(yesterday, datetime.time.max)
    response = client.get("/diary", params={"user_id": user_id, "until": until_dt.isoformat()})
    assert response.status_code == 200
    diaries_until = response.json()["diaries"]
    assert all(datetime.date.fromisoformat(d["diary_date"]) <= yesterday for d in diaries_until)

    # Update today's diary content and metadata (upsert via POST)
    update_payload = {
        "user_id": user_id,
        "diary_date": today.isoformat(),
        "content": "Today diary updated",
        "metadata": {"mood": "great"},
    }
    response = client.post("/diary", json=update_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "diary upserted"

    response = client.get("/diary", params={"user_id": user_id, "diary_date": today.isoformat()})
    assert response.status_code == 200
    diaries_after_update = response.json()["diaries"]
    assert diaries_after_update[0]["content"] == "Today diary updated"
    assert diaries_after_update[0]["metadata"].get("mood") == "great"

    # Upsert again to confirm update of existing entry
    second_update_payload = {
        "user_id": user_id,
        "diary_date": today.isoformat(),
        "content": "Today diary second update",
        "metadata": {"mood": "awesome"},
    }
    response = client.post("/diary", json=second_update_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "diary upserted"

    response = client.get("/diary", params={"user_id": user_id, "diary_date": today.isoformat()})
    assert response.status_code == 200
    diaries_after_second = response.json()["diaries"]
    assert len(diaries_after_second) == 1
    assert diaries_after_second[0]["content"] == "Today diary second update"
    assert diaries_after_second[0]["metadata"].get("mood") == "awesome"

    # Delete specific and all
    response = client.delete("/diary", params={"user_id": user_id, "diary_date": today.isoformat()})
    assert response.status_code == 200
    assert response.json()["status"] == "diary deleted"

    response = client.get("/diary", params={"user_id": user_id})
    assert response.status_code == 200
    remaining = response.json()["diaries"]
    assert all(datetime.date.fromisoformat(d["diary_date"]) != today for d in remaining)

    response = client.delete("/diary", params={"user_id": user_id})
    assert response.status_code == 200
    assert response.json()["status"] == "diary deleted"

    response = client.get("/diary", params={"user_id": user_id})
    assert response.status_code == 200
    assert response.json()["diaries"] == []

def test_summary_endpoints(test_user, test_session):
    # First, add some conversation history so a summary can be generated.
    payload = {
        "user_id": test_user,
        "session_id": test_session,
        "messages": [
            {"role": "user", "content": "How is the weather?", "metadata": {}},
            {"role": "assistant", "content": "It is sunny.", "metadata": {}}
        ]
    }
    response = client.post("/history", json=payload)
    assert response.status_code == 200

    # POST /summary/create: Create summary for the session.
    response = client.post("/summary/create", params={"user_id": test_user, "session_id": test_session})
    assert response.status_code == 200
    assert response.json()["status"] == "summary created"

    # GET /summary: Retrieve the generated summary.
    response = client.get("/summary", params={"user_id": test_user, "session_id": test_session})
    assert response.status_code == 200
    data = response.json()
    assert "summaries" in data
    assert len(data["summaries"]) >= 1

def test_knowledge_endpoints(test_user):
    # POST /knowledge: Add a knowledge record.
    payload = {
        "user_id": test_user,
        "knowledge": "Test knowledge content"
    }
    response = client.post("/knowledge", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "knowledge added"

    # GET /knowledge: Retrieve knowledge records.
    response = client.get("/knowledge", params={"user_id": test_user})
    assert response.status_code == 200
    data = response.json()
    assert "knowledge" in data
    assert len(data["knowledge"]) >= 1

def test_search_endpoint(test_user, test_session, monkeypatch):
    # Patch embed/llm to avoid external calls and make deterministic answers.
    fake_embedding = [0.01] * chat_memory.embedding_dimension

    async def fake_embed(text: str):
        return fake_embedding

    async def fake_llm(system_prompt: str, user_prompt: str):
        return "fake answer"

    monkeypatch.setattr(chat_memory, "embed", fake_embed)
    monkeypatch.setattr(chat_memory, "llm", fake_llm)

    # Ensure there is conversation history and a summary for search.
    payload = {
        "user_id": test_user,
        "session_id": test_session,
        "messages": [
            {"role": "user", "content": "Tell me a joke", "metadata": {}},
            {"role": "assistant", "content": "Why did the chicken cross the road?", "metadata": {}}
        ]
    }
    response = client.post("/history", json=payload)
    assert response.status_code == 200

    # Create summary for the session.
    response = client.post("/summary/create", params={"user_id": test_user, "session_id": test_session})
    assert response.status_code == 200

    # Add a diary entry to verify diaries are included in retrieval.
    today = datetime.date.today().isoformat()
    diary_payload = {
        "user_id": test_user,
        "diary_date": today,
        "content": "Diary note about a joke",
        "metadata": {"mood": "fun"},
    }
    response = client.post("/diary", json=diary_payload)
    assert response.status_code == 200

    # POST /search: Perform a search query.
    search_payload = {
        "user_id": test_user,
        "query": "joke",
        "top_k": 3,
        "search_content": False,
        "include_retrieved_data": True
    }
    response = client.post("/search", json=search_payload)
    assert response.status_code == 200
    data = response.json()
    # The response should include a "result" field containing an "answer".
    assert "result" in data
    result = data["result"]
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert "Diary" in result.get("retrieved_data", "")
    assert "Diary note about a joke" in result.get("retrieved_data", "")

    # Cleanup: Delete created history and summaries.
    response = client.delete("/history", params={"user_id": test_user, "session_id": test_session})
    assert response.status_code == 200
    response = client.delete("/summary", params={"user_id": test_user, "session_id": test_session})
    assert response.status_code == 200
    response = client.delete("/diary", params={"user_id": test_user})
    assert response.status_code == 200

def test_search_endpoint_multi_user(monkeypatch):
    """search endpoint should accept multiple user_ids (OR)."""
    fake_embedding = [0.02] * chat_memory.embedding_dimension

    async def fake_embed(text: str):
        return fake_embedding

    async def fake_llm(system_prompt: str, user_prompt: str):
        return "multi user api answer"

    monkeypatch.setattr(chat_memory, "embed", fake_embed)
    monkeypatch.setattr(chat_memory, "llm", fake_llm)

    user_a = str(uuid.uuid4())
    user_b = str(uuid.uuid4())
    session_a = f"session_{uuid.uuid4()}"
    session_b = f"session_{uuid.uuid4()}"

    payload_a = {
        "user_id": user_a,
        "session_id": session_a,
        "messages": [
            {"role": "user", "content": "User A loves pizza", "metadata": {}},
            {"role": "assistant", "content": "Noted", "metadata": {}},
        ],
    }
    payload_b = {
        "user_id": user_b,
        "session_id": session_b,
        "messages": [
            {"role": "user", "content": "User B likes pasta", "metadata": {}},
            {"role": "assistant", "content": "Sounds great", "metadata": {}},
        ],
    }
    assert client.post("/history", json=payload_a).status_code == 200
    assert client.post("/history", json=payload_b).status_code == 200

    assert client.post("/summary/create", params={"user_id": user_a, "session_id": session_a}).status_code == 200
    assert client.post("/summary/create", params={"user_id": user_b, "session_id": session_b}).status_code == 200

    today = datetime.date.today().isoformat()
    assert client.post("/diary", json={"user_id": user_a, "diary_date": today, "content": "Pizza diary"}).status_code == 200
    assert client.post("/diary", json={"user_id": user_b, "diary_date": today, "content": "Pasta diary"}).status_code == 200

    search_payload = {
        "user_id": [user_a, user_b],
        "query": "pasta",
        "top_k": 2,
        "search_content": False,
        "include_retrieved_data": True,
    }
    response = client.post("/search", json=search_payload)
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert isinstance(data["result"]["answer"], str)
    assert "Pasta" or "pasta" or "User B" in data["result"].get("retrieved_data", "")

    # Cleanup
    client.delete("/history", params={"user_id": user_a, "session_id": session_a})
    client.delete("/history", params={"user_id": user_b, "session_id": session_b})
    client.delete("/summary", params={"user_id": user_a, "session_id": session_a})
    client.delete("/summary", params={"user_id": user_b, "session_id": session_b})
    client.delete("/diary", params={"user_id": user_a})
    client.delete("/diary", params={"user_id": user_b})
