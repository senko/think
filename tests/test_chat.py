import pytest

from think.chat import Chat


def test_chat_constructor_without_content():
    chat = Chat()
    assert chat.messages == []


def test_chat_constructor_with_content():
    chat = Chat("Hello, world!")
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "system"
    assert chat.messages[0]["content"] == "Hello, world!"


def test_chat_constructor_with_whitespace_content():
    chat = Chat("  Hello, world!  ")
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "system"
    assert chat.messages[0]["content"] == "Hello, world!"


def test_add_unknown_role_raises_value_error():
    chat = Chat()
    with pytest.raises(ValueError) as excinfo:
        chat.add("unknown", "hello")
    assert str(excinfo.value) == "Unknown role: unknown"


def test_add_adds_message_with_role_and_content():
    chat = Chat()
    chat.add("user", "hello")
    assert chat.messages[0] == {"role": "user", "content": "hello"}


def test_add_adds_message_with_role_content_and_name():
    chat = Chat()
    chat.add("user", "hello", "Alice")
    assert chat.messages[0] == {"role": "user", "content": "hello", "name": "Alice"}


def test_add_dedents_string_content():
    chat = Chat()
    chat.add("user", "\n    hello\n    world\n")
    assert chat.messages[0]["content"] == "\nhello\nworld"


def test_add_forwards_dict_content():
    chat = Chat()
    chat.add("user", {"text": "hello"})
    assert chat.messages[0]["content"] == {"text": "hello"}


def test_system_adds_system_message():
    chat = Chat()
    chat.system("Hello, world!")
    assert chat.messages == [{"role": "system", "content": "Hello, world!"}]


def test_system_adds_system_message_with_name():
    chat = Chat()
    chat.system("Hello, world!", "System")
    assert chat.messages == [
        {"role": "system", "content": "Hello, world!", "name": "System"}
    ]


def test_system_dedents_content():
    chat = Chat()
    chat.system("    Hello, world!")
    assert chat.messages == [{"role": "system", "content": "Hello, world!"}]


def test_system_preserves_lines_in_content():
    chat = Chat()
    chat.system("Hello,\nworld!")
    assert chat.messages == [{"role": "system", "content": "Hello,\nworld!"}]


def test_user_adds_user_message():
    chat = Chat()
    chat.user("Hello, World!")
    assert chat.messages[0] == {"role": "user", "content": "Hello, World!"}


def test_user_adds_user_message_with_name():
    chat = Chat()
    chat.user("Hello, World!", "John Doe")
    assert chat.messages[0] == {
        "role": "user",
        "content": "Hello, World!",
        "name": "John Doe",
    }


def test_user_raises_error_if_content_is_empty_string():
    chat = Chat()
    with pytest.raises(ValueError):
        chat.user("")


def test_user_raises_error_if_content_is_none():
    chat = Chat()
    with pytest.raises(ValueError):
        chat.user(None)


def test_assistant_adds_correct_message():
    chat = Chat()
    chat.assistant("Hello, world!")
    assert chat.messages == [{"role": "assistant", "content": "Hello, world!"}]


def test_assistant_dedents_content():
    chat = Chat()
    chat.assistant("    Hello, world!")
    assert chat.messages == [{"role": "assistant", "content": "Hello, world!"}]


def test_assistant_adds_name_if_provided():
    chat = Chat()
    chat.assistant("Hello, world!", "Geppetto")
    assert chat.messages == [
        {"role": "assistant", "content": "Hello, world!", "name": "Geppetto"}
    ]


def test_assistant_returns_self():
    chat = Chat()
    result = chat.assistant("Hello, world!")
    assert result is chat


def test_function_message_added_correctly():
    chat = Chat()
    chat.function("Hello World")
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "function"
    assert chat.messages[0]["content"] == "Hello World"


def test_function_message_with_name_added_correctly():
    chat = Chat()
    chat.function("Hello World", name="Function1")
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "function"
    assert chat.messages[0]["content"] == "Hello World"
    assert chat.messages[0]["name"] == "Function1"


def test_function_message_content_dedented():
    chat = Chat()
    chat.function("  Hello World  ")
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "function"
    assert chat.messages[0]["content"] == "Hello World"


def test_function_message_return_chat_object():
    chat = Chat()
    result = chat.function("Hello World")
    assert isinstance(result, Chat)


def test_function_message_with_empty_content():
    chat = Chat()
    with pytest.raises(ValueError):
        chat.function("")


def test_function_message_with_non_string_content():
    chat = Chat()
    with pytest.raises(TypeError):
        chat.function(123)


def test_chat_fork():
    chat1 = Chat("Hello")
    chat1.user("Hello LLM!")
    chat1.assistant("Hello User!")
    chat2 = chat1.fork()
    assert chat1.messages == chat2.messages
    chat1.user("New message in chat1")
    assert chat1.messages != chat2.messages
    chat2.assistant("New message in chat2")
    assert chat1.messages != chat2.messages


def test_chat_fork_with_no_messages():
    chat1 = Chat()
    chat2 = chat1.fork()
    assert chat1.messages == chat2.messages
    chat1.user("New message in chat1")
    assert chat1.messages != chat2.messages
    chat2.assistant("New message in chat2")
    assert chat1.messages != chat2.messages


def test_chat_fork_with_multiple_messages():
    chat1 = Chat("Init")
    chat1.user("Hello!").assistant("Hi!").user("How are you?")
    chat2 = chat1.fork()
    assert chat1.messages == chat2.messages
    chat1.assistant("I'm good! How are you?")
    assert chat1.messages != chat2.messages
    chat2.assistant("I'm fine! How are you?")
    assert chat1.messages != chat2.messages


def test_after_with_empty_chats():
    chat1 = Chat()
    chat2 = Chat()
    new_chat = chat1.after(chat2)
    assert new_chat.messages == []


def test_after_with_no_common_messages():
    chat1 = Chat()
    chat1.user("Hello")
    chat2 = Chat()
    chat2.user("Hi")
    new_chat = chat1.after(chat2)
    assert new_chat.messages == chat1.messages


def test_after_with_some_common_messages():
    chat1 = Chat()
    chat1.user("Hello").assistant("How can I assist?")
    chat2 = chat1.fork()
    chat2.user("What's the weather?")
    new_chat = chat2.after(chat1)
    assert new_chat.messages == [{"role": "user", "content": "What's the weather?"}]


def test_after_with_all_common_messages():
    chat1 = Chat()
    chat1.user("Hello")
    chat2 = chat1.fork()
    new_chat = chat2.after(chat1)
    assert new_chat.messages == []


def test_after_with_more_messages_in_parent_chat():
    chat1 = Chat()
    chat1.user("Hello").assistant("How can I assist?").user("What's the weather?")
    chat2 = Chat()
    chat2.user("Hello").assistant("How can I assist?")
    new_chat = chat1.after(chat2)
    assert new_chat.messages == [{"role": "user", "content": "What's the weather?"}]


def test_last_empty_chat():
    chat = Chat()
    assert chat.last() is None


def test_last_single_message_chat():
    chat = Chat()
    chat.user("Hello")
    assert chat.last()["content"] == "Hello"


def test_last_multiple_messages_chat():
    chat = Chat()
    chat.user("Hello")
    chat.assistant("Hi")
    assert chat.last()["content"] == "Hi"


def test_last_after_fork():
    chat = Chat()
    chat.user("Hello")
    forked_chat = chat.fork()
    forked_chat.assistant("Hi")
    assert chat.last()["content"] == "Hello"
    assert forked_chat.last()["content"] == "Hi"


def test_last_after_deepcopy():
    from copy import deepcopy

    chat = Chat()
    chat.user("Hello")
    copied_chat = deepcopy(chat)
    copied_chat.assistant("Hi")
    assert chat.last()["content"] == "Hello"
    assert copied_chat.last()["content"] == "Hi"


def test_message_iterator():
    chat = Chat("hello").user("world")
    messages = []
    for message in chat:
        messages.append(message)
    assert messages == [
        {"role": "system", "content": "hello"},
        {"role": "user", "content": "world"},
    ]
