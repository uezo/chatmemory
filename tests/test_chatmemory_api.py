import os
import uuid
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from chatmemory import ChatMemory


# Create a ChatMemory instance with real configuration.
DB_NAME = os.getenv("DB_NAME", "your_db")
DB_USER = os.getenv("DB_USER", "your_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 5432)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

chat_memory = ChatMemory(
    openai_api_key=OPENAI_API_KEY,
    openai_base_url=OPENAI_BASE_URL,
    llm_model=LLM_MODEL,
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

def test_search_endpoint(test_user, test_session):
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

    # Cleanup: Delete created history and summaries.
    response = client.delete("/history", params={"user_id": test_user, "session_id": test_session})
    assert response.status_code == 200
    response = client.delete("/summary", params={"user_id": test_user, "session_id": test_session})
    assert response.status_code == 200
