from datetime import datetime, timedelta
from fastapi import FastAPI, Depends
from logging import getLogger
from pydantic import BaseModel
import traceback
from typing import List, Dict
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import uvicorn
from chatmemory.chatmemory import ChatMemory

logger = getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


class HistoriesRequest(BaseModel):
    messages: List[Message]


class HistoriesResponse(BaseModel):
    messages: List[Message]


class Archive(BaseModel):
    date: str
    archive: str


class ArchivesResponse(BaseModel):
    archives: List[Archive]


class EntitiesResponse(BaseModel):
    entities: Dict[str, object]


class ApiResponse(BaseModel):
    message: str


class ChatMemoryServer:
    def __init__(self, openai_apikey: str, database_url: str="sqlite:///chatmemory.db"):
        self.database_url = database_url
        self.engine = create_engine(self.database_url)
        self.session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        self.openai_apikey = openai_apikey
        self.chatmemory = ChatMemory(api_key=self.openai_apikey)
        self.chatmemory.create_database(self.engine)

        self.app = FastAPI()
        self.setup_handlers()

    def get_db(self):
        db = self.session_local()
        try:
            yield db
        finally:
            db.close()

    def setup_handlers(self):
        app = self.app

        @app.post("/histories/{user_id}", response_model=ApiResponse)
        async def add_histories(user_id: str, request: HistoriesRequest, db: Session = Depends(self.get_db)):
            try:
                self.chatmemory.add_histories(
                    db, user_id,
                    [{"role": m.role, "content": m.content} for m in request.messages]
                )
                db.commit()
                return ApiResponse(message="Histories added successfully")
            
            except Exception as ex:
                logger.error(f"Error at add_histories: {ex}\n{traceback.format_exc()}")
                return ApiResponse(message="Error")

        @app.get("/histories/{user_id}", response_model=HistoriesResponse)
        async def get_histories(user_id: str, since: str=None, until: str=None, db: Session = Depends(self.get_db)):
            histories = self.chatmemory.get_histories(
                db, user_id,
                datetime.strptime(since, "%Y-%m-%d") if since else None,
                datetime.strptime(until, "%Y-%m-%d") if until else None
            )
            return HistoriesResponse(messages=[
                Message(role=h["role"], content=h["content"])
                for h in histories
            ])

        @app.post("/archives/{user_id}", response_model=ApiResponse)
        async def archive_histories(user_id: str, target_date: str=None, days: int=1, db: Session = Depends(self.get_db)):
            try:
                for i in range(days):
                    self.chatmemory.archive_histories(
                        db, user_id,
                        (datetime.strptime(target_date, "%Y-%m-%d") if target_date else datetime.utcnow()).date() - timedelta(days=days - i - 1),
                    )
                    db.commit()
                return ApiResponse(message="Histories archived successfully")

            except Exception as ex:
                logger.error(f"Error at archive_histories: {ex}\n{traceback.format_exc()}")
                return ApiResponse(message="Error")

        @app.get("/archives/{user_id}", response_model=ArchivesResponse)
        async def get_archives(user_id: str, since: str=None, until: str=None, db: Session = Depends(self.get_db)):
            archives = self.chatmemory.get_archives(
                db, user_id,
                datetime.strptime(since, "%Y-%m-%d") if since else None,
                datetime.strptime(until, "%Y-%m-%d") if until else None
            )
            return ArchivesResponse(archives=[
                Archive(date=a["date"].strftime("%Y-%m-%d"), archive=a["archive"])
                for a in archives
            ])


        @app.post("/entities/{user_id}", response_model=ApiResponse)
        async def parse_entities(user_id: str, target_date: str=None, days: int=1, db: Session = Depends(self.get_db)):
            try:
                for i in range(days):
                    self.chatmemory.parse_entities(
                        db, user_id,
                        (datetime.strptime(target_date, "%Y-%m-%d") if target_date else datetime.utcnow()).date() - timedelta(days=days - i - 1),
                    )
                    db.commit()
                return ApiResponse(message="Entities parsed and stored successfully")

            except Exception as ex:
                logger.error(f"Error at parse_entities: {ex}\n{traceback.format_exc()}")
                return ApiResponse(message="Error")        

        @app.get("/entities/{user_id}", response_model=EntitiesResponse)
        async def get_entities(user_id: str, db: Session = Depends(self.get_db)):
            entities = self.chatmemory.get_entities(db, user_id)
            return EntitiesResponse(entities=entities)


        @app.delete("/all/{user_id}", response_model=ApiResponse)
        async def delete_all(user_id: str, db: Session = Depends(self.get_db)):
            try:
                self.chatmemory.delete(db, user_id)
                db.commit()
                return ApiResponse(message=f"Delete all data for {user_id} successfully")

            except Exception as ex:
                logger.error(f"Error at delete_all: {ex}\n{traceback.format_exc()}")
                return ApiResponse(message="Error")        

    def start(self, host :str="127.0.0.1", port: int=8123):
        uvicorn.run(self.app, host=host, port=port)
