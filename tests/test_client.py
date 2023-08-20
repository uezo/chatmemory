import pytest
from chatmemory.client import ChatMemoryClient
from unittest.mock import patch, Mock

@pytest.fixture
def client():
    return ChatMemoryClient()

def test_get_archived_histories_content(client):
    with patch("requests.get", return_value=Mock(json=lambda: {"archives": [{"date": "2023-08-19", "archive": "test_archive"}]})):
        result = client.get_archived_histories_content("test_user")
        expected_content = (
            "以下はここ3日間にあったユーザーとの会話を要約したものです。\n"
            "```\n"
            "- 2023-08-19: test_archive\n\n"
            "```\n\n"
            "基本的にこの会話の情報を利用する必要はありませんが、会話の流れでこれらの情報が必要になった場合は利用してください。\n"
        )
        assert result == expected_content

def test_set_archived_histories_message(client):
    # No system message
    with patch("chatmemory.client.ChatMemoryClient.get_archived_histories_content", return_value="test_content"):
        # 1st turn
        messages = []
        client.set_archived_histories_message("test_user", messages)
        assert messages == []
        # 2nd turn
        messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hi there!"}]
        client.set_archived_histories_message("test_user", messages)
        assert messages == [{"role": "user", "content": "test_content"}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hi there!"}]

    # With system message
    with patch("chatmemory.client.ChatMemoryClient.get_archived_histories_content", return_value="test_content"):
        # 1st turn
        messages = [{"role": "system", "content": "system_content"}]
        client.set_archived_histories_message("test_user", messages)
        assert messages == [{"role": "system", "content": "system_content"}]
        # 2nd turn
        messages = [{"role": "system", "content": "system_content"}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hi there!"}]
        client.set_archived_histories_message("test_user", messages)
        assert messages == [{"role": "system", "content": "system_content"}, {"role": "user", "content": "test_content"}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hi there!"}]

    # Custom injection point
    with patch("chatmemory.client.ChatMemoryClient.get_archived_histories_content", return_value="test_content"):
        client.archive_injection_at = 3
        # 1st turn
        messages = [{"role": "system", "content": "system_content"}]
        client.set_archived_histories_message("test_user", messages)
        assert messages == [{"role": "system", "content": "system_content"}]
        # 2nd turn
        messages = [{"role": "system", "content": "system_content"}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hi there!"}]
        client.set_archived_histories_message("test_user", messages)
        assert messages == [{"role": "system", "content": "system_content"}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hi there!"}]
        # 3rd turn
        messages = [{"role": "system", "content": "system_content"}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hi there!"}, {"role": "user", "content": "Hi2"}, {"role": "assistant", "content": "Hi there!2"}]
        client.set_archived_histories_message("test_user", messages)
        assert messages == [{"role": "system", "content": "system_content"}, {"role": "user", "content": "test_content"}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hi there!"}, {"role": "user", "content": "Hi2"}, {"role": "assistant", "content": "Hi there!2"}]

def test_get_entities_content(client):
    with patch("requests.get", return_value=Mock(json=lambda: {"name": "John"})):
        result = client.get_entities_content("test_user")
        expected_content = (
            "\n\n"
            "# ユーザーに関して会話を通じて聞き出したこと\n\n"
            "以下はユーザーとの会話を通じてあなたが記憶している内容です。強く意識する必要はありませんが、会話の流れでこれらの情報が必要になった場合はこれらの情報を会話に利用してください。\n\n"
            "```\n"
            "{\n"
            "  \"name\": \"John\"\n"
            "}\n"
            "```\n"
        )
        assert result == expected_content
