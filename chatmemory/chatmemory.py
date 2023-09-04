from datetime import datetime, date, time, timedelta, timezone
import json
from logging import getLogger, NullHandler
from sqlalchemy import Column, Integer, String, DateTime, Date
from sqlalchemy.orm import Session, declarative_base
from openai import ChatCompletion

logger = getLogger(__name__)
logger.addHandler(NullHandler())


# Models
Base = declarative_base()

class History(Base):
    __tablename__ = "histories"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String)
    role = Column(String)
    content = Column(String)
    

class Archive(Base):
    __tablename__ = "archives"
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, primary_key=True, index=True)
    archive_date = Column(DateTime, primary_key=True, index=True)
    archive = Column(String)


class Entity(Base):
    __tablename__ = "entities"
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, primary_key=True, index=True)
    last_target_date = Column(Date, nullable=False)
    serialized_entities = Column(String)


# Archiver
class HistoryArchiver:
    def __init__(self, api_key: str, model: str="gpt-3.5-turbo-16k-0613", archive_length: int=100):
        self.api_key = api_key
        self.model = model
        self.archive_length = archive_length

    def archive(self, messages: list):
        histories_text = ""
        for m in messages:
            if m["role"] == "user" or m["role"] == "assistant":
                histories_text += f'- {m["role"]}: {m["content"]}\n'

        histories = [
            {"role": "user", "content": f"以下の会話の内容を、話題等に注目して{self.archive_length}文字以内程度で要約してください。要約した文章は第三者視点で、主語はuserとasssitantとします。\n\n{histories_text}"}
        ]

        functions = [{
            "name": "save_summarized_histories",
            "description": "会話の内容を話題に注目して要約して保存する",
            "parameters": {
                "type": "object",
                "properties": {
                    "summarized_text": {
                        "type": "string",
                        "description": "要約した会話の内容"
                    }
                },
                "required": ["summarized_text"]
            }
        }]

        resp = ChatCompletion.create(
            api_key=self.api_key,
            model=self.model,
            messages=histories,
            functions=functions,
            function_call={"name": "save_summarized_histories"}
        )

        return json.loads(resp["choices"][0]["message"]["function_call"]["arguments"])["summarized_text"]


# Parser
class EntityParser:
    def __init__(self, api_key: str, model: str="gpt-3.5-turbo-16k-0613"):
        self.api_key = api_key
        self.model = model

    def parse(self, messages: list):
        histories = [m for m in messages if m["role"] == "user" or m["role"] == "assistant"]
        histories.append({"role": "user", "content": "会話の履歴の中から、ユーザーに関して覚えておくべき情報があれば抽出してください。nameは英語かつsnake_caseで表現します。"})

        functions = [{
            "name": "save_entities",
            "description": "ニックネームや誕生日、所在地、好きなものや嫌いなものなどユーザーに関する情報を保存する",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "name of entity. use snake case.", "examples": ["birthday_date"]},
                                "value": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }]

        resp = ChatCompletion.create(
            api_key=self.api_key,
            model=self.model,
            messages=histories,
            functions=functions,
            function_call={"name": "save_entities"}
        )

        return {
            e["name"]: e["value"] for e
            in json.loads(
                resp["choices"][0]["message"]["function_call"]["arguments"]
            )["entities"]
        }


# Memory manager
class ChatMemory:
    def __init__(self, api_key: str=None, model: str="gpt-3.5-turbo-16k-0613", history_archiver: HistoryArchiver=None, entity_parser: EntityParser=None):
        self.history_archiver = history_archiver or HistoryArchiver(api_key, model)
        self.entity_parser = entity_parser or EntityParser(api_key, model)
        self.history_max_count = 100
        self.archive_retrive_count = 5

    def today(self) -> datetime:
        return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    def date_to_utc_datetime(self, d) -> datetime:
        return datetime.combine(d, time()).replace(tzinfo=timezone.utc)

    def create_database(self, engine):
        Base.metadata.create_all(bind=engine)

    def add_histories(self, session: Session, user_id: str, messages: list):
        histories = [
            History(user_id=user_id, role=m["role"], content=m["content"])
            for m in messages if m["role"] == "user" or m["role"] == "assistant"
        ]
        session.bulk_save_objects(histories)

    def get_histories(self, session: Session, user_id: str, since: datetime=None, until: datetime=None) -> list:
        histories = session.query(History).filter(
            History.user_id == user_id,
            History.timestamp >= (since or datetime.min),
            History.timestamp <= (until or datetime.max)
        ).order_by(History.id).limit(self.history_max_count).all()

        return [{"role": h.role, "content": h.content} for h in histories]

    def archive_histories(self, session: Session, user_id: str, target_date: datetime=None):
        target_date = target_date or self.today()
        conversation_history = self.get_histories(session, user_id, target_date, target_date + timedelta(days=1))
        
        if conversation_history:
            summarized_archive = self.history_archiver.archive(conversation_history)
            
            new_archive_entry = Archive(
                timestamp=datetime.utcnow(),
                user_id=user_id,
                archive_date=target_date,
                archive=summarized_archive
            )
            
            session.merge(new_archive_entry)
        
        else:
            logger.info("No histories found to archive")

    def get_archives(self, session: Session, user_id: str, since: datetime=None, until: datetime=None) -> list:
        archives = session.query(Archive.archive_date, Archive.archive).filter(
            Archive.user_id == user_id,
            Archive.archive_date >= (since or datetime.min),
            Archive.archive_date <= (until or datetime.max)
        ).order_by(Archive.archive_date.desc()).limit(self.archive_retrive_count).all()

        return [{ "date": a.archive_date, "archive": a.archive } for a in archives]

    def parse_entities(self, session: Session, user_id: str, target_date: date):
        # Get histories on target_date
        since_dt = self.date_to_utc_datetime(target_date)
        until_dt = since_dt + timedelta(days=1)
        conversation_history = self.get_histories(session, user_id, since_dt, until_dt)
        if len(conversation_history) == 0:
            logger.info(f"No histories found on {target_date} for parsing entities")
            return

        # Get stored entities or new entities
        stored_entites = session.query(Entity).filter(
            Entity.user_id == user_id,
        ).first() or Entity(user_id=user_id, last_target_date=date.min)

        # Skip parsing if already parsed
        if stored_entites.last_target_date >= target_date:
            if datetime.utcnow().date() != target_date:
                logger.info(f"Entities in histories on {target_date} are already parsed")
                return

        entities = self.entity_parser.parse(conversation_history)

        if stored_entites.serialized_entities:
            entities_json = json.loads(stored_entites.serialized_entities)
            for k, v in entities.items():
                entities_json[k] = v
        else:
            entities_json = entities
        stored_entites.timestamp = datetime.utcnow()
        stored_entites.serialized_entities = json.dumps(entities_json, ensure_ascii=False)
        stored_entites.last_target_date = target_date

        session.merge(stored_entites)

    def get_entities(self, session: Session, user_id: str) -> dict:
        entities = session.query(Entity).filter(
            Entity.user_id == user_id,
        ).first()

        if entities and entities.serialized_entities:
            return json.loads(entities.serialized_entities)
        else:
            return {}

    def delete(self, session: Session, user_id: str):
        session.query(History).filter(History.user_id == user_id).delete()
        session.query(Archive).filter(Archive.user_id == user_id).delete()
        session.query(Entity).filter(Entity.user_id == user_id).delete()
