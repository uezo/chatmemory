import datetime
import json
import logging
from contextlib import contextmanager
from time import sleep
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
import psycopg2
from psycopg2 import pool
import openai

logger = logging.getLogger(__name__)


# ==============================
# Data models
# ==============================

class HistoryMessage(BaseModel):
    created_at: Optional[datetime.datetime] = None  # If not provided, server sets the timestamp
    role: str
    content: str
    metadata: dict = {}

class HistoryMessageWithId(HistoryMessage):
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class SessionSummary(BaseModel):
    created_at: datetime.datetime
    session_id: str
    summary: str

class Knowledge(BaseModel):
    created_at: datetime.datetime
    knowledge: str

class KnowledgeWithIds(Knowledge):
    id: int
    user_id: str

class SearchResult(BaseModel):
    answer: str
    retrieved_data: Optional[str] = None


# ==============================
# API Schemas
# ==============================

class AddHistoryRequest(BaseModel):
    user_id: str
    session_id: str
    messages: List[HistoryMessage]

class AddHistoryResponse(BaseModel):
    status: str

class GetHistoryResponse(BaseModel):
    messages: List[HistoryMessageWithId]

class GetSessionIdsResponse(BaseModel):
    session_ids: List[str]

class CreateSummaryResponse(BaseModel):
    status: str

class GetSummaryResponse(BaseModel):
    summaries: List[SessionSummary]

class AddKnowledgeRequest(BaseModel):
    user_id: str
    knowledge: str

class GetKnowledgeResponse(BaseModel):
    knowledge: List[KnowledgeWithIds]

class SearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = 5
    search_content: bool = False
    include_retrieved_data: bool = False

class SearchResponse(BaseModel):
    result: SearchResult

class DeleteResponse(BaseModel):
    status: str


# ==============================
# Custom Exception
# ==============================

class ChatMemoryError(Exception):
    """Custom exception for ChatMemory errors."""
    pass


# ==============================
# ChatMemory Class
# ==============================

class ChatMemory:
    SEARCH_SYSTEM_PROMPT_DEFAULT = (
        "Based on the following information, provide a concise and accurate answer to the user's question. "
        "If there is conflicting information, prioritize the information with the later timestamp. "
        "The answer should be based solely on the provided information and should not include any personal opinions or follow-up questions."
    )
    SEARCH_USER_PROMPT_DEFAULT = ""
    SEARCH_USER_PROMPT_CONTENT_DEFAULT = (
        "If the above information is sufficient, provide an answer based on it. "
        "If not, please append [search:content] to your answer."
    )
    SUMMARISE_SYSTEM_PROMPT_DEFAULT = "Summarize the following conversation and include useful keywords for search."

    def __init__(
        self,
        *,
        openai_api_key: str = None,
        openai_base_url: str = None,
        llm_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        db_name: str = "chatmemory",
        db_user: str = "postgres",
        db_password: str = "postgres",
        db_host: str = "127.0.0.1",
        db_port: int = 5432,
        search_system_prompt: str = None,
        search_user_prompt: str = None,
        search_user_prompt_content: str = None,
        summarize_system_prompt: str = None,
    ):
        # Initialize the asynchronous OpenAI client
        self.openai_client = openai.AsyncOpenAI(
            api_key=openai_api_key, base_url=openai_base_url
        )
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.search_system_prompt = search_system_prompt or self.SEARCH_SYSTEM_PROMPT_DEFAULT
        self.search_user_prompt = search_user_prompt or self.SEARCH_USER_PROMPT_DEFAULT
        self.search_user_prompt_content = search_user_prompt_content or self.SEARCH_USER_PROMPT_CONTENT_DEFAULT
        self.summarize_system_prompt = summarize_system_prompt or self.SUMMARISE_SYSTEM_PROMPT_DEFAULT

        # Database configuration dictionary
        self.db_config = {
            "dbname": db_name,
            "user": db_user,
            "password": db_password,
            "host": db_host,
            "port": db_port,
        }
        self.connection_pool = pool.SimpleConnectionPool(1, 10, **self.db_config)
        logger.info("Initializing database...")
        retry_counter = 2
        while True:
            try:
                self.init_db()
                break
            except Exception as ex:
                if "Connection refused" in str(ex) and retry_counter >= 0:
                    logger.warning("Retrying database connection...")
                    sleep(2)
                    retry_counter -= 1
                else:
                    raise ex

    @contextmanager
    def get_db_cursor(self):
        """
        Context manager for database connection and cursor.
        Automatically commits changes or rolls back on exception,
        and always closes the connection.
        """
        conn = self.connection_pool.getconn()
        try:
            # Connection health check
            try:
                with conn.cursor() as test_cursor:
                    test_cursor.execute("SELECT 1")
            except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
                logger.warning("Discarding zombie connection.")
                self.connection_pool.putconn(conn, close=True)
                conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            yield cursor, conn
            conn.commit()
        except Exception as ex:
            conn.rollback()
            logger.error(f"Database error: {ex}")
            raise ChatMemoryError(ex)
        finally:
            cursor.close()
            if conn.closed:
                self.connection_pool.putconn(conn, close=True)
            else:
                self.connection_pool.putconn(conn)


    def init_db(self):
        """Initialize necessary tables, extensions, and indexes."""
        with self.get_db_cursor() as (cur, _):
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_history_user_id ON conversation_history(user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_history_session_id ON conversation_history(session_id);")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    embedding_summary VECTOR(1536),
                    content_embedding VECTOR(1536)
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_summary_user_id ON conversation_summaries(user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_summary_session_id ON conversation_summaries(session_id);")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_knowledge (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL,
                    user_id TEXT NOT NULL,
                    knowledge TEXT NOT NULL,
                    embedding VECTOR(1536)
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_user_id ON user_knowledge(user_id);")
            logger.info("Database initialized successfully.")

    async def llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM with given prompts and return the answer."""
        try:
            resp = await self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as ex:
            logger.error(f"Error in LLM: {ex}")
            raise ChatMemoryError(ex)

    async def embed(self, text: str) -> List[float]:
        """Call the embedding API for a given text and return the embedding vector."""
        try:
            resp = await self.openai_client.embeddings.create(
                model=self.embedding_model, input=text
            )
            return resp.data[0].embedding
        except Exception as ex:
            logger.error(f"Error in embedding: {ex}")
            raise ChatMemoryError(ex)

    def add_history(self, user_id: str, session_id: str, messages: List[HistoryMessage]):
        """Insert multiple conversation history records for a given user and session."""
        now = datetime.datetime.utcnow()
        with self.get_db_cursor() as (cur, _):
            for msg in messages:
                created_at = msg.created_at or now
                cur.execute(
                    """
                    INSERT INTO conversation_history (created_at, user_id, session_id, role, content, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (created_at, user_id, session_id, msg.role, msg.content, json.dumps(msg.metadata)),
                )
        logger.info(f"Inserted {len(messages)} messages for user {user_id}, session {session_id}.")

    def get_history(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[HistoryMessageWithId]:
        """
        Retrieve conversation history for a given user or session (up to 1000 records).
        At least one of user_id or session_id must be provided.
        """
        if not user_id and not session_id:
            raise ValueError("Either user_id or session_id must be specified.")
        with self.get_db_cursor() as (cur, _):
            if user_id and session_id:
                cur.execute(
                    """
                    SELECT created_at, user_id, session_id, role, content, metadata
                    FROM conversation_history
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY created_at ASC
                    LIMIT 1000
                    """,
                    (user_id, session_id),
                )
            elif session_id:
                cur.execute(
                    """
                    SELECT created_at, user_id, session_id, role, content, metadata
                    FROM conversation_history
                    WHERE session_id = %s
                    ORDER BY created_at ASC
                    LIMIT 1000
                    """,
                    (session_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT created_at, user_id, session_id, role, content, metadata
                    FROM conversation_history
                    WHERE user_id = %s
                    ORDER BY created_at ASC
                    LIMIT 1000
                    """,
                    (user_id,),
                )
            rows = cur.fetchall()
        logger.info(f"Retrieved {len(rows)} history records.")
        return [HistoryMessageWithId(
            created_at=row[0],
            user_id=row[1],
            session_id=row[2],
            role=row[3],
            content=row[4],
            metadata=row[5]
        ) for row in rows]

    def delete_history(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Delete conversation history. If both user_id and session_id are provided, delete records matching both."""
        with self.get_db_cursor() as (cur, _):
            if user_id and session_id:
                cur.execute(
                    "DELETE FROM conversation_history WHERE user_id = %s AND session_id = %s",
                    (user_id, session_id),
                )
                logger.info(f"Deleted history for user {user_id} and session {session_id}.")
            elif session_id:
                cur.execute(
                    "DELETE FROM conversation_history WHERE session_id = %s", (session_id,)
                )
                logger.info(f"Deleted history for session {session_id}.")
            elif user_id:
                cur.execute(
                    "DELETE FROM conversation_history WHERE user_id = %s", (user_id,)
                )
                logger.info(f"Deleted history for user {user_id}.")
            else:
                cur.execute("DELETE FROM conversation_history")
                logger.info("Deleted all conversation history.")

    def get_session_ids(self, user_id: str) -> List[str]:
        """Retrieve all distinct session IDs for the specified user."""
        with self.get_db_cursor() as (cur, _):
            cur.execute(
                """
                SELECT DISTINCT session_id
                FROM conversation_history
                WHERE user_id = %s
                ORDER BY session_id ASC
                """,
                (user_id,),
            )
            session_ids = [row[0] for row in cur.fetchall()]
        logger.info(f"Retrieved {len(session_ids)} session IDs for user {user_id}.")
        return session_ids

    async def create_summary(self, user_id: str, session_id: str = None, overwrite: bool = False, max_count: int = 10):
        """
        Generate and store a summary (with embeddings) for a given session.
        If a summary already exists and overwrite is False, this method does nothing.
        """
        if session_id:
            session_ids = [session_id]
        else:
            session_ids = self.get_session_ids(user_id)
            session_ids = session_ids[(max_count + 1) * -1:-1]  # Skip the latest session

        for session_id in session_ids:
            is_summary_exists = False
            # Check if summary already exists
            with self.get_db_cursor() as (cur, _):
                cur.execute(
                    """
                    SELECT id FROM conversation_summaries
                    WHERE user_id = %s AND session_id = %s
                    """,
                    (user_id, session_id),
                )
                if cur.fetchone() is not None:
                    is_summary_exists = True

            if is_summary_exists and not overwrite:
                logger.info(f"Summary already exists for user {user_id}, session {session_id}.")
                return

            # Retrieve conversation history for the session
            with self.get_db_cursor() as (cur, _):
                cur.execute(
                    """
                    SELECT role, content FROM conversation_history
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY created_at ASC
                    """,
                    (user_id, session_id),
                )
                messages = cur.fetchall()
            logger.info(f"Retrieved {len(messages)} messages for summary generation (user {user_id}, session {session_id}).")

            conversation_text = "\n".join([f"{role}: {content}" for role, content in messages])
            try:
                summary = await self.llm(self.summarize_system_prompt, conversation_text)
                logger.info(f"Generated summary for user {user_id}, session {session_id}.")
            except ChatMemoryError:
                logger.error("Summary generation failed.")
                return

            try:
                embedding_summary = await self.embed(summary)
                embedding_content = await self.embed(conversation_text)
            except ChatMemoryError:
                logger.error("Embedding for summary or content failed.")
                return

            with self.get_db_cursor() as (cur, _):
                if is_summary_exists:
                    cur.execute(
                        """
                        UPDATE conversation_summaries
                        SET created_at = %s,
                            summary = %s,
                            embedding_summary = %s,
                            content_embedding = %s
                        WHERE user_id = %s AND session_id = %s
                        """,
                        (datetime.datetime.utcnow(), summary, embedding_summary, embedding_content, user_id, session_id),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO conversation_summaries (created_at, user_id, session_id, summary, embedding_summary, content_embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (datetime.datetime.utcnow(), user_id, session_id, summary, embedding_summary, embedding_content),
                    )
            logger.info(f"Summary stored for user {user_id}, session {session_id}.")

    def get_summaries(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[SessionSummary]:
        """
        Retrieve conversation summaries for a specified user or session (up to 1000 records).
        If both are provided, use an AND condition.
        """
        if not user_id and not session_id:
            raise ValueError("Either user_id or session_id must be specified.")
        with self.get_db_cursor() as (cur, _):
            if user_id and session_id:
                cur.execute(
                    """
                    SELECT created_at, session_id, summary
                    FROM conversation_summaries
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY created_at ASC
                    LIMIT 1000
                    """,
                    (user_id, session_id),
                )
            elif session_id:
                cur.execute(
                    """
                    SELECT created_at, session_id, summary
                    FROM conversation_summaries
                    WHERE session_id = %s
                    ORDER BY created_at ASC
                    LIMIT 1000
                    """,
                    (session_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT created_at, session_id, summary
                    FROM conversation_summaries
                    WHERE user_id = %s
                    ORDER BY created_at ASC
                    LIMIT 1000
                    """,
                    (user_id,),
                )
            rows = cur.fetchall()
        logger.info(f"Retrieved {len(rows)} summaries.")
        return [SessionSummary(created_at=row[0], session_id=row[1], summary=row[2]) for row in rows]

    def delete_summaries(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Delete conversation summaries. If both user_id and session_id are provided, delete records matching both."""
        with self.get_db_cursor() as (cur, _):
            if user_id and session_id:
                cur.execute(
                    "DELETE FROM conversation_summaries WHERE user_id = %s AND session_id = %s",
                    (user_id, session_id),
                )
                logger.info(f"Deleted summaries for user {user_id} and session {session_id}.")
            elif session_id:
                cur.execute(
                    "DELETE FROM conversation_summaries WHERE session_id = %s", (session_id,)
                )
                logger.info(f"Deleted summaries for session {session_id}.")
            elif user_id:
                cur.execute(
                    "DELETE FROM conversation_summaries WHERE user_id = %s", (user_id,)
                )
                logger.info(f"Deleted summaries for user {user_id}.")
            else:
                cur.execute("DELETE FROM conversation_summaries")
                logger.info("Deleted all summaries.")

    async def add_knowledge(self, user_id: str, knowledge: str):
        """
        Generate an embedding for the given knowledge text and store it.
        """
        now = datetime.datetime.utcnow()
        embedding = await self.embed(knowledge)
        logger.info("Knowledge embedding generated.")
        with self.get_db_cursor() as (cur, _):
            cur.execute(
                """
                INSERT INTO user_knowledge (created_at, user_id, knowledge, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (now, user_id, knowledge, embedding),
            )
        logger.info(f"Knowledge added for user {user_id}.")

    def get_knowledge(self, user_id: str) -> List[KnowledgeWithIds]:
        """Retrieve all knowledge records for a given user."""
        with self.get_db_cursor() as (cur, _):
            cur.execute(
                """
                SELECT id, created_at, user_id, knowledge
                FROM user_knowledge
                WHERE user_id = %s
                ORDER BY created_at ASC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
        logger.info(f"Retrieved {len(rows)} knowledge records for user {user_id}.")
        return [KnowledgeWithIds(id=row[0], created_at=row[1], user_id=row[2], knowledge=row[3]) for row in rows]

    def delete_knowledge(self, user_id: str, knowledge_id: Optional[int] = None):
        """
        Delete knowledge for a given user. If knowledge_id is provided, delete that record;
        otherwise, delete all knowledge for the user.
        """
        with self.get_db_cursor() as (cur, _):
            if knowledge_id:
                cur.execute("DELETE FROM user_knowledge WHERE id = %s AND user_id = %s", (knowledge_id, user_id))
                logger.info(f"Deleted knowledge record {knowledge_id} for user {user_id}.")
            else:
                cur.execute("DELETE FROM user_knowledge WHERE user_id = %s", (user_id,))
                logger.info(f"Deleted all knowledge for user {user_id}.")

    def search_summary(self, cur, user_id: str, query_embedding_str: str, top_k: int) -> List[SessionSummary]:
        """
        Search for conversation summaries using vector similarity.
        """
        cur.execute(
            """
            SELECT session_id, summary, created_at
            FROM conversation_summaries
            WHERE user_id = %s
            ORDER BY embedding_summary <-> %s::vector
            LIMIT %s
            """,
            (user_id, query_embedding_str, top_k),
        )
        rows = cur.fetchall()
        return [SessionSummary(created_at=row[2], session_id=row[0], summary=row[1]) for row in rows]

    def search_knowledge(self, cur, user_id: str, query_embedding_str: str, top_k: int) -> List[Knowledge]:
        """
        Search for user knowledge records using vector similarity.
        """
        cur.execute(
            """
            SELECT created_at, knowledge
            FROM user_knowledge
            WHERE user_id = %s
            ORDER BY embedding <-> %s::vector
            LIMIT %s
            """,
            (user_id, query_embedding_str, top_k),
        )
        rows = cur.fetchall()
        return [Knowledge(created_at=row[0], knowledge=row[1]) for row in rows]

    def search_content(self, conn, user_id: str, query_embedding_str: str, top_k: int) -> Dict[str, List[HistoryMessage]]:
        """
        Search conversation histories based on summaries.
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT session_id
                FROM conversation_summaries
                WHERE user_id = %s
                ORDER BY embedding_summary <-> %s::vector
                LIMIT %s
                """,
                (user_id, query_embedding_str, top_k),
            )
            session_ids = [r[0] for r in cur.fetchall()]

        session_messages = {}
        for session_id in session_ids:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT created_at, role, content
                    FROM conversation_history
                    WHERE session_id = %s
                    ORDER BY created_at ASC
                    LIMIT 100
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
                session_messages[session_id] = [HistoryMessage(created_at=r[0], role=r[1], content=r[2]) for r in rows]
        return session_messages

    async def search(self, user_id: str, query: str, top_k: int = 5, search_content: bool = False, include_retrieved_data: bool = False) -> SearchResult:
        """
        Search conversation summaries and user knowledge based on a query,
        and generate an answer using the LLM.
        """
        query_embedding = await self.embed(query)
        vector_str = "[" + ",".join(map(str, query_embedding)) + "]"

        with self.get_db_cursor() as (cur, conn):
            summaries = self.search_summary(cur, user_id, vector_str, top_k)
            summaries_text = "\n".join(
                [f"Conversation summary ({s.created_at}): {s.summary}" for s in summaries]
            ) if summaries else ""

            knowledges = self.search_knowledge(cur, user_id, vector_str, top_k)
            knowledges_text = "\n".join(
                [f"Knowledge about user ({k.created_at}): {k.knowledge}" for k in knowledges]
            ) if knowledges else ""

            retrieved_data = ""
            if summaries_text or knowledges_text:
                if summaries_text:
                    retrieved_data += f"====\n\n{summaries_text}\n\n"
                if knowledges_text:
                    retrieved_data += f"====\n\n{knowledges_text}\n\n"

                user_prompt = f"User Question: {query}\n\n{retrieved_data}\n====\n\n"
                user_prompt += (self.search_user_prompt_content if search_content else self.search_user_prompt)

                answer = await self.llm(self.search_system_prompt, user_prompt)

                if "[search:content]" not in answer:
                    return SearchResult(answer=answer, retrieved_data=retrieved_data if include_retrieved_data else None)
                else:
                    logger.info("Insufficient information; proceeding to search conversation content.")
            else:
                logger.info("No summaries or knowledge found; proceeding to search conversation content.")

            # Fallback: search conversation history content
            content_data = self.search_content(conn, user_id, vector_str, top_k)
            content_retrieved = "====\n"
            for session_id, messages in content_data.items():
                if messages:
                    content_retrieved += f"\n- Conversation log ({messages[0].created_at}):\n"
                    for m in messages:
                        content_retrieved += f"  - {m.role}: {m.content}\n"
            retrieved_data += content_retrieved

            answer = await self.llm(self.search_system_prompt, f"User Question: {query}\n\n{retrieved_data}\n====\n\n{self.search_user_prompt}")
            return SearchResult(answer=answer, retrieved_data=retrieved_data if include_retrieved_data else None)

    def get_router(self, prefix: str = None) -> APIRouter:
        """
        Return an APIRouter with all API endpoints configured.
        This allows users to include the router in their FastAPI app.
        """
        router = APIRouter(prefix=prefix or "")

        # ----- History Endpoints -----
        @router.post("/history", response_model=AddHistoryResponse, summary="Add conversation history records", tags=["History"])
        async def post_history_endpoint(request: AddHistoryRequest, background_tasks: BackgroundTasks):
            """
            Add multiple conversation history records.
            If a session switch is detected, generate the summary for the previous session in the background.
            """
            try:
                # Get the latest session_id for the user before insertion
                with self.get_db_cursor() as (cur, _):
                    cur.execute(
                        """
                        SELECT session_id FROM conversation_history
                        WHERE user_id = %s
                        ORDER BY id DESC LIMIT 1
                        """,
                        (request.user_id,),
                    )
                    row = cur.fetchone()
                    previous_session = row[0] if row else None

                self.add_history(user_id=request.user_id, session_id=request.session_id, messages=request.messages)
                # If a session switch is detected, schedule summary generation for the previous session
                if previous_session and previous_session != request.session_id:
                    background_tasks.add_task(self.create_summary, request.user_id, previous_session)
                return AddHistoryResponse(status="ok")
            except Exception as ex:
                logger.error(f"Error in post_history_endpoint: {ex}")
                raise HTTPException(status_code=500, detail=str(ex))

        @router.get("/history", response_model=GetHistoryResponse, summary="Get conversation history", tags=["History"])
        def get_history_endpoint(user_id: Optional[str] = Query(None), session_id: Optional[str] = Query(None)):
            """
            Retrieve conversation history for a specified user or session (max 1000 records).
            """
            try:
                messages = self.get_history(user_id=user_id, session_id=session_id)
                return GetHistoryResponse(messages=messages)
            except ValueError as ve:
                logger.error(f"Get history error: {ve}")
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as ex:
                logger.error(f"Error in get_history_endpoint: {ex}")
                raise HTTPException(status_code=500, detail=str(ex))

        @router.delete("/history", response_model=DeleteResponse, summary="Delete conversation history", tags=["History"])
        def delete_history_endpoint(user_id: Optional[str] = Query(None), session_id: Optional[str] = Query(None)):
            """
            Delete conversation history for a specified user, session, or all history if not specified.
            """
            try:
                self.delete_history(user_id=user_id, session_id=session_id)
                return DeleteResponse(status="history deleted")
            except Exception as ex:
                logger.error(f"Error in delete_history_endpoint: {ex}")
                raise HTTPException(status_code=500, detail=str(ex))

        @router.get("/history/session_ids", response_model=GetSessionIdsResponse, summary="Get user sessions", tags=["History"])
        def get_sessions_endpoint(user_id: str = Query(...)):
            """
            Retrieve all distinct session IDs for the specified user.
            """
            try:
                session_ids = self.get_session_ids(user_id)
                return GetSessionIdsResponse(session_ids=session_ids)
            except Exception as ex:
                logger.error(f"Error in get_sessions_endpoint: {ex}")
                raise HTTPException(status_code=500, detail=str(ex))

        # ----- Summary Endpoints -----
        @router.post("/summary/create", response_model=CreateSummaryResponse, summary="Create summary for a session or sessions of the user", tags=["Summary"])
        async def create_summary_endpoint(user_id: str, session_id: str = None, overwrite: bool = False, max_count: int = 10):
            """
            Generate and store a summary (and embeddings) for the specified session or sessions of the specified user.
            """
            try:
                await self.create_summary(user_id, session_id, overwrite, max_count)
                return CreateSummaryResponse(status="summary created")
            except Exception as ex:
                logger.error(f"Error in create_summary_endpoint: {ex}")
                raise HTTPException(status_code=500, detail=str(ex))

        @router.get("/summary", response_model=GetSummaryResponse, summary="Get conversation summaries", tags=["Summary"])
        def get_summary_endpoint(user_id: Optional[str] = Query(None), session_id: Optional[str] = Query(None)):
            """
            Retrieve conversation summaries for a specified user or session (max 1000 records).
            """
            try:
                summaries = self.get_summaries(user_id=user_id, session_id=session_id)
                return GetSummaryResponse(summaries=summaries)
            except ValueError as ve:
                logger.error(f"Get summary error: {ve}")
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as ex:
                logger.error(f"Error in get_summary_endpoint: {ex}")
                raise HTTPException(status_code=500, detail=str(ex))

        @router.delete("/summary", response_model=DeleteResponse, summary="Delete conversation summaries", tags=["Summary"])
        def delete_summary_endpoint(user_id: Optional[str] = Query(None), session_id: Optional[str] = Query(None)):
            """
            Delete conversation summaries for a specified user, session, or all if not provided.
            """
            try:
                self.delete_summaries(user_id=user_id, session_id=session_id)
                return DeleteResponse(status="summaries deleted")
            except Exception as ex:
                logger.error(f"Error in delete_summary_endpoint: {ex}")
                raise HTTPException(status_code=500, detail=str(ex))

        # ----- Knowledge Endpoints -----
        @router.post("/knowledge", response_model=DeleteResponse, summary="Add user knowledge", tags=["Knowledge"])
        async def add_knowledge_endpoint(request: AddKnowledgeRequest):
            """
            Add a knowledge record for the specified user.
            """
            try:
                await self.add_knowledge(request.user_id, request.knowledge)
                return DeleteResponse(status="knowledge added")
            except Exception as ex:
                logger.error(f"Error in add_knowledge_endpoint: {ex}")
                raise HTTPException(status_code=500, detail=str(ex))

        @router.get("/knowledge", response_model=GetKnowledgeResponse, summary="Get user knowledge", tags=["Knowledge"])
        def get_knowledge_endpoint(user_id: str = Query(...)):
            """
            Retrieve all knowledge records for the specified user.
            """
            try:
                knowledge_records = self.get_knowledge(user_id)
                return GetKnowledgeResponse(knowledge=knowledge_records)
            except Exception as ex:
                logger.error(f"Error in get_knowledge_endpoint: {ex}")
                raise HTTPException(status_code=500, detail=str(ex))

        @router.delete("/knowledge", response_model=DeleteResponse, summary="Delete user knowledge", tags=["Knowledge"])
        def delete_knowledge_endpoint(user_id: str = Query(...), knowledge_id: Optional[int] = Query(None)):
            """
            Delete knowledge for the specified user. If knowledge_id is provided, delete that record;
            otherwise, delete all knowledge for the user.
            """
            try:
                self.delete_knowledge(user_id, knowledge_id)
                return DeleteResponse(status="knowledge deleted")
            except Exception as ex:
                logger.error(f"Error in delete_knowledge_endpoint: {ex}")
                raise HTTPException(status_code=500, detail=str(ex))

        # ----- Search Endpoint -----
        @router.post("/search", response_model=SearchResponse, summary="Search conversation summaries and user knowledge, and generate answer", tags=["Search"])
        async def search_endpoint(request: SearchRequest):
            """
            Search both conversation summaries and user knowledge for similar records,
            and generate an answer using the LLM.
            """
            try:
                result = await self.search(
                    request.user_id, request.query, request.top_k, request.search_content, request.include_retrieved_data
                )
                return SearchResponse(result=result)
            except Exception as ex:
                logger.error(f"Error in search_endpoint: {ex}")
                raise HTTPException(status_code=500, detail=str(ex))

        return router

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from fastapi import FastAPI
    import uvicorn

    load_dotenv()

    DB_NAME = os.getenv("DB_NAME", "chatmemory")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT = os.getenv("DB_PORT", 5432)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHATMEMORY_PORT = os.getenv("CHATMEMORY_PORT", 8000)

    cm = ChatMemory(
        openai_api_key=OPENAI_API_KEY,
        openai_base_url=OPENAI_BASE_URL,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        db_name=DB_NAME,
        db_user=DB_USER,
        db_password=DB_PASSWORD,
        db_port=DB_PORT,
    )

    # Start server
    app = FastAPI(title="ChatMemory", version="0.2.1")
    app.include_router(cm.get_router())
    uvicorn.run(app, host="0.0.0.0", port=int(CHATMEMORY_PORT))
