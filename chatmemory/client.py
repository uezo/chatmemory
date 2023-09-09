import json
import requests

class ChatMemoryClient:
    def __init__(self, url: str="http://127.0.0.1:8123", archive_injection_at: int=2):
        self.url = url
        self.archive_injection_at = archive_injection_at

    def add_histories(self, user_id: str, messages: list):
        requests.post(f"{self.url}/histories/{user_id}", json={"messages": messages})

    def get_archived_histories(self, user_id: str) -> list:
        return requests.get(f"{self.url}/archives/{user_id}").json()["archives"]

    def get_archived_histories_content(self, user_id: str) -> str:
        archives = self.get_archived_histories(user_id)
        if archives:
            archives_str = ""
            for a in archives:
                archives_str += f'- {a["date"]}: {a["archive"]}\n'

            return f"""以下はここ3日間にあったユーザーとの会話を要約したものです。
```
{archives_str}
```

基本的にこの会話の情報を利用する必要はありませんが、会話の流れでこれらの情報が必要になった場合は利用してください。
"""

        return ""

    def archive(self, user_id: str, target_date: str=None, days: int=None):
        data = {}
        if target_date: data["target_date"] = target_date
        if days: data["days"] = days

        return requests.post(f"{self.url}/archives/{user_id}", json=data).json()

    def set_archived_histories_message(self, user_id: str, messages: list):
        if len([m for m in messages if m["role"] == "assistant"]) == self.archive_injection_at - 1:
            archived_histories_content = self.get_archived_histories_content(user_id)
            if archived_histories_content:
                messages.insert(
                    1 if messages[0]["role"] == "system" else 0,
                    {"role": "user", "content": archived_histories_content}
                )

    def get_entities(self, user_id: str) -> dict:
        return requests.get(f"{self.url}/entities/{user_id}").json()

    def get_entities_content(self, user_id: str) -> str:
        entities = self.get_entities(user_id)

        if entities:
            return f"""

# ユーザーに関して会話を通じて聞き出したこと

以下はユーザーとの会話を通じてあなたが記憶している内容です。強く意識する必要はありませんが、会話の流れでこれらの情報が必要になった場合はこれらの情報を会話に利用してください。

```
{json.dumps(entities, ensure_ascii=False, indent=2)}
```
"""
        
        return ""

    def extract_entities(self, user_id: str, target_date: str=None, days: int=None):
        data = {}
        if target_date: data["target_date"] = target_date
        if days: data["days"] = days

        return requests.post(f"{self.url}/entities/{user_id}", json=data).json()
