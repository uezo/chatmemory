import os
from datetime import datetime, date, time, timedelta, timezone
import base64
import json
import hashlib
from logging import getLogger, NullHandler
import traceback
from sqlalchemy import Column, Integer, String, DateTime, Date
from sqlalchemy.orm import Session, declarative_base
from openai import ChatCompletion
from Crypto.Cipher import AES


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
    archive_date = Column(Date, primary_key=True, index=True)
    archive = Column(String)


class Entity(Base):
    __tablename__ = "entities"
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, primary_key=True, index=True)
    last_target_date = Column(Date, nullable=False)
    serialized_entities = Column(String)


# Archiver
class HistoryArchiver:
    PROMPT_EN = "Please summarize the content of the following conversation in the original language of the content(e.g. content in Japanese should be summarize in Japanese), in about {archive_length} words, paying attention to the topics discussed. Write the summary in third-person perspective, with 'user' and 'assistant' as the subjects.\n\n{histories_text}"
    PROMPT_JA = "以下の会話の内容を、話題等に注目して{archive_length}文字以内程度の日本語で要約してください。要約した文章は第三者視点で、主語はuserとasssitantとします。\n\n{histories_text}"

    def __init__(self, api_key: str, model: str="gpt-3.5-turbo-16k-0613", archive_length: int=100, prompt: str=PROMPT_EN):
        self.api_key = api_key
        self.model = model
        self.archive_length = archive_length
        self.archive_prompt = prompt

    def archive(self, messages: list):
        histories_text = ""
        for m in messages:
            if m["role"] == "user" or m["role"] == "assistant":
                histories_text += f'- {m["role"]}: {m["content"]}\n'

        histories = [
            {"role": "user", "content": self.archive_prompt.format(archive_length=self.archive_length, histories_text=histories_text)}
        ]

        functions = [{
            "name": "save_summarized_histories",
            "description": "Summarize the content of the conversation.",
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

        try:
            return json.loads(resp["choices"][0]["message"]["function_call"]["arguments"])["summarized_text"]

        except json.decoder.JSONDecodeError:
            logger.warning(f"Retry parsing JSON: {resp}")
            jstr = resp["choices"][0]["message"]["function_call"]["arguments"].replace("\",\n}", "\"\n}")
            return json.loads(jstr)["summarized_text"]

        except Exception as ex:
            logger.error(f"Invalid response form ChatGPT at archive: {resp}\n{ex}\n{traceback.format_exc()}")
            raise ex


class EntityExtractor:
    PROMPT_EN = "From the conversation history, please extract any information that should be remembered about the user in original language. If there are already stored items, overwrite the new information with the same item key."
    PROMPT_JA = "会話の履歴の中から、ユーザーに関して覚えておくべき情報があれば抽出してください。既に記憶している項目があれば、同じ項目名を使用して新しい情報で上書きします。"

    def __init__(self, api_key: str, model: str="gpt-3.5-turbo-16k-0613", prompt: str=PROMPT_EN):
        self.api_key = api_key
        self.model = model
        self.extract_prompt = prompt

    def extract(self, messages: list, entities: dict=None):
        histories = [m for m in messages if m["role"] == "user" or m["role"] == "assistant"]

        prompt = self.extract_prompt
        if entities:
            prompt = self.extract_prompt + "\n\nEntities that you already know:\n"
            for k, v in entities.items():
                prompt += f"- {k}: {v}\n"

        histories.append({"role": "user", "content": prompt})

        functions = [{
            "name": "save_entities",
            "description": "Extract and save any information that should be remembered about the user.",
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

        try:
            return {
                e["name"]: e["value"] for e
                in json.loads(
                    resp["choices"][0]["message"]["function_call"]["arguments"]
                )["entities"]
            }
        
        except json.decoder.JSONDecodeError:
            logger.warning(f"Retry parsing JSON: {resp}")
            jstr = resp["choices"][0]["message"]["function_call"]["arguments"].replace("\",\n}", "\"\n}")
            return {
                e["name"]: e["value"] for e in json.loads(jstr)["entities"]
            }

        except Exception as ex:
            logger.error(f"Invalid response form ChatGPT at extract: {resp}\n{ex}\n{traceback.format_exc()}")
            raise ex


# Memory manager
class ChatMemory:
    def __init__(self, api_key: str=None, model: str="gpt-3.5-turbo-16k-0613", history_archiver: HistoryArchiver=None, entity_extractor: EntityExtractor=None):
        self.history_archiver = history_archiver or HistoryArchiver(api_key, model)
        self.entity_extractor = entity_extractor or EntityExtractor(api_key, model)
        self.history_max_count = 100
        self.archive_retrive_count = 5

    def date_to_utc_datetime(self, d) -> datetime:
        return datetime.combine(d, time()).replace(tzinfo=timezone.utc)

    def encrypt(self, text: str, password: str=None):
        if not password:
            return text

        salt = os.urandom(16)
        key = hashlib.scrypt(password=password.encode("utf-8"), salt=salt, n=2**5, r=8, p=1, dklen=32)
        cipher = AES.new(key, AES.MODE_GCM)
        cipher_text, tag = cipher.encrypt_and_digest(text.encode("utf-8"))
        return "-".join([base64.b64encode(item).decode("utf-8") for item in [salt, cipher.nonce, cipher_text, tag]])

    def decrypt(self, encrypted_text: str, password: str=None):
        if not password:
            return encrypted_text

        salt, cipher_nonce, cipher_text, tag = [base64.b64decode(item) for item in encrypted_text.split("-")]
        key = hashlib.scrypt(password=password.encode("utf-8"), salt=salt, n=2**5, r=8, p=1, dklen=32)
        cipher = AES.new(key, AES.MODE_GCM, cipher_nonce)
        return cipher.decrypt_and_verify(cipher_text, tag).decode("utf-8")

    def create_database(self, engine):
        Base.metadata.create_all(bind=engine)

    def add_histories(self, session: Session, user_id: str, messages: list, password: str=None):
        histories = [
            History(user_id=user_id, role=m["role"], content=self.encrypt(m["content"], password))
            for m in messages if m["role"] == "user" or m["role"] == "assistant"
        ]
        session.bulk_save_objects(histories)

    def get_histories(self, session: Session, user_id: str, since: datetime=None, until: datetime=None, password: str=None) -> list:
        histories = session.query(History).filter(
            History.user_id == user_id,
            History.timestamp >= (since or datetime.min),
            History.timestamp <= (until or datetime.max)
        ).order_by(History.id).limit(self.history_max_count).all()

        return [{"role": h.role, "content": self.decrypt(h.content, password)} for h in histories]

    def archive_histories(self, session: Session, user_id: str, target_date: date, password: str=None):
        since_dt = self.date_to_utc_datetime(target_date)
        conversation_history = self.get_histories(
            session,
            user_id,
            since_dt,
            since_dt + timedelta(days=1),
            password
        )

        if len(conversation_history) == 0:
            logger.info(f"No histories found on {target_date} to archive")
            return

        # Get stored archive
        stored_archive = session.query(Archive).filter(
            Archive.user_id == user_id,
            Archive.archive_date == target_date
        ).first() or Archive(
            user_id=user_id,
            timestamp=datetime.min,
            archive_date=target_date
        )

        # Skip if already archived
        if stored_archive:
            if stored_archive.timestamp.date() > target_date:
                logger.info(f"Histories on {target_date} are already archived")
                return

        summarized_archive = self.history_archiver.archive(conversation_history)

        stored_archive.timestamp = datetime.utcnow()
        stored_archive.archive = self.encrypt(summarized_archive, password)

        session.merge(stored_archive)

    def get_archives(self, session: Session, user_id: str, since: date=None, until: date=None, password: str=None) -> list:
        archives = session.query(Archive.archive_date, Archive.archive).filter(
            Archive.user_id == user_id,
            Archive.archive_date >= (since or date.min),
            Archive.archive_date <= (until or date.max)
        ).order_by(Archive.archive_date.desc()).limit(self.archive_retrive_count).all()

        return [{ "date": a.archive_date, "archive": self.decrypt(a.archive, password) } for a in archives]

    def extract_entities(self, session: Session, user_id: str, target_date: date, password: str=None):
        # Get histories on target_date
        since_dt = self.date_to_utc_datetime(target_date)
        until_dt = since_dt + timedelta(days=1)
        conversation_history = self.get_histories(session, user_id, since_dt, until_dt, password)
        if len(conversation_history) == 0:
            logger.info(f"No histories found on {target_date} for extracting entities")
            return

        # Get stored entities or new entities
        stored_entites = session.query(Entity).filter(
            Entity.user_id == user_id,
        ).first() or Entity(user_id=user_id, last_target_date=date.min)

        # Skip extraction if already extracted (larger than target_date because some histories on last_target_date may be not processed)
        if stored_entites.last_target_date > target_date:
            logger.info(f"Entities in histories on {target_date} are already extracted")
            return

        if stored_entites.serialized_entities:
            entities_json = json.loads(self.decrypt(stored_entites.serialized_entities, password))
        else:
            entities_json = {}

        new_entities = self.entity_extractor.extract(conversation_history, entities_json)
        for k, v in new_entities.items():
            entities_json[k] = v

        stored_entites.timestamp = datetime.utcnow()
        stored_entites.serialized_entities = self.encrypt(json.dumps(entities_json, ensure_ascii=False), password)
        stored_entites.last_target_date = target_date

        session.merge(stored_entites)

    def get_entities(self, session: Session, user_id: str, password: str=None) -> dict:
        entities = session.query(Entity).filter(
            Entity.user_id == user_id,
        ).first()

        if entities and entities.serialized_entities:
            return json.loads(self.decrypt(entities.serialized_entities, password))
        else:
            return {}

    def delete(self, session: Session, user_id: str):
        session.query(History).filter(History.user_id == user_id).delete()
        session.query(Archive).filter(Archive.user_id == user_id).delete()
        session.query(Entity).filter(Entity.user_id == user_id).delete()
