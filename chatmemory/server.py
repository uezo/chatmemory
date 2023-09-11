from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, Header
from fastapi.responses import JSONResponse
from logging import getLogger
from pydantic import BaseModel, Field
import traceback
from typing import List, Dict, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import uvicorn
from .chatmemory import ChatMemory

logger = getLogger(__name__)


class Message(BaseModel):
    role: str = Field(..., title="role", description="The role of the author of this message.", example="user")
    content: Optional[str] = Field(None, title="content", description="The contents of the message.", example="Hello!")


class HistoriesRequest(BaseModel):
    messages: List[Message] = Field(..., title="messages", description="A list of messages to store comprising the conversation so far.", example=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}])


class HistoriesResponse(BaseModel):
    messages: List[Message] = Field(..., title="messages", description="A list of retrieved messages comprising the conversation so far.", example=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}])

class ArchivesRequest(BaseModel):
    target_date: Optional[str] = Field(None, title="target_date", description="Target date in ISO8601 format for creating the summary of conversation.", example="2023-08-11")
    days: int = Field(1, title="days", description="The number of days to go back in the conversation history for creating an archive.", example=1)


class Archive(BaseModel):
    date: str = Field(..., title="date", description="Date in ISO8601 format", example="2023-08-11")
    archive: str = Field(..., title="archive", description="Summarized text of the conversation on the date.", example="user")


class ArchivesResponse(BaseModel):
    archives: List[Archive] = Field(..., title="archives", description="A list of summarized conversation texts.", example=[{"date": "2023-08-11", "archive": "User and assistant talk about lunch and user says that soba is nice."}, {"date": "2023-08-10", "archive": "User says she loves cats."}])


class EntitiesRequest(BaseModel):
    target_date: Optional[str] = Field(None, title="target_date", description="Target date in ISO8601 format to extract entities.", example="2023-08-11")
    days: int = Field(1, title="days", description="The number of days to go back in the conversation history to extract entities.", example=1)
    entities: Optional[Dict[str, object]] = Field(None, title="entities", description="Entities to store. All existing entities are replaced with this.", example={"nickname": "uezo", "age": 28, "favorite_food": "soba"})


class EntitiesResponse(BaseModel):
    entities: Dict[str, object] = Field(..., title="entities", description="Stored entities.", example={"nickname": "uezo", "age": 28, "favorite_food": "soba"})


class ApiResponse(BaseModel):
    message: str = Field(..., title="message", description="Message from API", example="Entities extracted and stored successfully")


class ChatMemoryServer:
    def __init__(self, openai_apikey: str, database_url: str="sqlite:///chatmemory.db", server_args: dict=None):
        self.database_url = database_url
        self.engine = create_engine(self.database_url)
        self.session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        self.openai_apikey = openai_apikey
        self.chatmemory = ChatMemory(api_key=self.openai_apikey)
        self.chatmemory.create_database(self.engine)

        self.app = FastAPI(**(server_args or {"title": "ChatMemory", "version": "0.1.3"}))
        self.setup_handlers()

    def get_db(self):
        db = self.session_local()
        try:
            yield db
        finally:
            db.close()

    def setup_handlers(self):
        app = self.app

        @app.post("/histories/{user_id}", response_model=ApiResponse, tags=["History"])
        async def add_histories(user_id: str, request: HistoriesRequest, encryption_key: str = Header(default=None), db: Session = Depends(self.get_db)):
            try:
                self.chatmemory.add_histories(
                    db, user_id,
                    [{"role": m.role, "content": m.content} for m in request.messages],
                    encryption_key
                )
                db.commit()
                return ApiResponse(message="Histories added successfully")
            
            except Exception as ex:
                logger.error(f"Error at add_histories: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)

        @app.get("/histories/{user_id}", response_model=HistoriesResponse, tags=["History"])
        async def get_histories(user_id: str, since: str=None, until: str=None, encryption_key: str = Header(default=None), db: Session = Depends(self.get_db)):
            try:
                histories = self.chatmemory.get_histories(
                    db, user_id,
                    datetime.strptime(since, "%Y-%m-%d") if since else None,
                    datetime.strptime(until, "%Y-%m-%d") if until else None,
                    encryption_key
                )
                return HistoriesResponse(messages=[
                    Message(role=h["role"], content=h["content"])
                    for h in histories
                ])

            except ValueError as verr:
                return JSONResponse("Invalid encryption key", 400)

            except Exception as ex:
                logger.error(f"Error at get_histories: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)


        @app.delete("/histories/{user_id}", response_model=ApiResponse, tags=["History"])
        async def delete_histories(user_id: str, db: Session = Depends(self.get_db)):
            try:
                self.chatmemory.delete_histories(db, user_id)
                db.commit()
                return ApiResponse(message="All histories are deleted successfully")

            except Exception as ex:
                logger.error(f"Error at delete_histories: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)

        @app.post("/archives/{user_id}", response_model=ApiResponse, tags=["Archive"])
        async def archive_histories(user_id: str, request: ArchivesRequest, encryption_key: str = Header(default=None), db: Session = Depends(self.get_db)):
            try:
                for i in range(request.days):
                    self.chatmemory.archive_histories(
                        db, user_id,
                        (datetime.strptime(request.target_date, "%Y-%m-%d") if request.target_date
                         else datetime.utcnow()).date() - timedelta(days=request.days - i - 1),
                        encryption_key
                    )
                    db.commit()
                return ApiResponse(message="Histories archived successfully")

            except Exception as ex:
                logger.error(f"Error at archive_histories: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)

        @app.get("/archives/{user_id}", response_model=ArchivesResponse, tags=["Archive"])
        async def get_archives(user_id: str, since: str=None, until: str=None, encryption_key: str = Header(default=None), db: Session = Depends(self.get_db)):
            try:
                archives = self.chatmemory.get_archives(
                    db, user_id,
                    datetime.strptime(since, "%Y-%m-%d") if since else None,
                    datetime.strptime(until, "%Y-%m-%d") if until else None,
                    encryption_key
                )
                return ArchivesResponse(archives=[
                    Archive(date=a["date"].strftime("%Y-%m-%d"), archive=a["archive"])
                    for a in archives
                ])

            except ValueError as verr:
                return JSONResponse("Invalid encryption key", 400)

            except Exception as ex:
                logger.error(f"Error at get_archives: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)


        @app.delete("/archives/{user_id}", response_model=ApiResponse, tags=["Archive"])
        async def delete_archives(user_id: str, db: Session = Depends(self.get_db)):
            try:
                self.chatmemory.delete_archives(db, user_id)
                db.commit()
                return ApiResponse(message="All archives are deleted successfully")

            except Exception as ex:
                logger.error(f"Error at delete_archives: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)

        @app.post("/entities/{user_id}", response_model=ApiResponse, tags=["Entity"])
        async def save_entities(user_id: str, request: EntitiesRequest, encryption_key: str = Header(default=None), db: Session = Depends(self.get_db)):
            try:
                now = datetime.utcnow()
                if request.entities is None:
                    for i in range(request.days):
                        self.chatmemory.extract_entities(
                            db, user_id,
                            (datetime.strptime(request.target_date, "%Y-%m-%d") if request.target_date
                            else now).date() - timedelta(days=request.days - i - 1),
                            encryption_key
                        )
                        db.commit()
                    return ApiResponse(message="Entities extracted and stored successfully")
            
                else:
                    self.chatmemory.save_entities(db, user_id, now, now.date(), request.entities, encryption_key)
                    db.commit()
                    return ApiResponse(message="Entities stored successfully")

            except Exception as ex:
                logger.error(f"Error at save_entities: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)


        @app.get("/entities/{user_id}", response_model=EntitiesResponse, tags=["Entity"])
        async def get_entities(user_id: str, encryption_key: str = Header(default=None), db: Session = Depends(self.get_db)):
            try:
                entities = self.chatmemory.get_entities(db, user_id, encryption_key)
                return EntitiesResponse(entities=entities)

            except ValueError as verr:
                return JSONResponse("Invalid encryption key", 400)

            except Exception as ex:
                logger.error(f"Error at get_entities: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)

        @app.delete("/entities/{user_id}", response_model=ApiResponse, tags=["Entity"])
        async def delete_entities(user_id: str, db: Session = Depends(self.get_db)):
            try:
                self.chatmemory.delete_entities(db, user_id)
                db.commit()
                return ApiResponse(message="All entities are deleted successfully")

            except Exception as ex:
                logger.error(f"Error at delete_entities: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)

        @app.delete("/all/{user_id}", response_model=ApiResponse, tags=["All"])
        async def delete_all(user_id: str, db: Session = Depends(self.get_db)):
            try:
                self.chatmemory.delete_all(db, user_id)
                db.commit()
                return ApiResponse(message=f"Delete all data for {user_id} successfully")

            except Exception as ex:
                logger.error(f"Error at delete_all: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)


    def start(self, host :str="127.0.0.1", port: int=8123):
        uvicorn.run(self.app, host=host, port=port)
