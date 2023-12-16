from copy import deepcopy
from typing import Iterator


class Chat:
    """
    A conversation between a user and a Large Language Model (LLM) assistant.
    """

    ROLES = ["system", "user", "assistant", "function"]

    messages: list[dict[str, str]]

    def __init__(self, content: str | None = None):
        """
        Initialize a new conversation.

        :param content: Initial system message (optional).
        """
        self.messages = []
        if content is not None:
            self.system(content)

    @staticmethod
    def _dedent(text: str) -> str:
        """
        Remove common leading whitespace from every line of text.

        :param text: Text to dedent.
        :returns: Dedented text.
        """
        indent = len(text)
        lines = text.splitlines()
        for line in lines:
            if line.strip():
                indent = min(indent, len(line) - len(line.lstrip()))
        dedented_lines = [line[indent:].rstrip() for line in lines]
        return "\n".join(line for line in dedented_lines)

    def add(self, role: str, content: str, name: str | None = None) -> "Chat":
        """
        Add a message to the conversation.

        In most cases, you should use the convenience methods instead.

        :param role: Role of the message (system, user, assistant, function).
        :param content: Content of the message.
        :param name: Name of the message sender (optional).
        :returns: The chat object.
        """

        if role not in self.ROLES:
            raise ValueError(f"Unknown role: {role}")
        if not content:
            raise ValueError("Empty message content")
        if not isinstance(content, str) and not isinstance(content, dict):
            raise TypeError(f"Invalid message content: {type(content).__name__}")

        message = {
            "role": role,
            "content": self._dedent(content) if isinstance(content, str) else content,
        }
        if name is not None:
            message["name"] = name

        self.messages.append(message)
        return self

    def system(self, content: str, name: str | None = None) -> "Chat":
        """
        Add a system message to the conversation.

        System messages can use `name` for showing example conversations
        between an example user and an example assistant.

        :param content: Content of the message.
        :param name: Name of the message sender (optional).
        :returns: The chat object.
        """
        return self.add("system", content, name)

    def user(self, content: str, name: str | None = None) -> "Chat":
        """
        Add a user message to the conversation.

        :param content: Content of the message.
        :param name: User name (optional).
        :returns: The chat object.
        """
        return self.add("user", content, name)

    def assistant(self, content: str, name: str | None = None) -> "Chat":
        """
        Add an assistant message to the conversation.

        :param content: Content of the message.
        :param name: Assistant name (optional).
        :returns: The chat object.
        """
        return self.add("assistant", content, name)

    def function(self, content: str, name: str | None = None) -> "Chat":
        """
        Add a function (tool) response to the conversation.

        :param content: Content of the message.
        :param name: Function/tool name (optional).
        :returns: The chat object.
        """
        return self.add("function", content, name)

    def fork(self) -> "Chat":
        """
        Create an identical copy of the conversation.

        This performs a deep copy of all the message
        contents, so you can safely modify both the
        parent and the child conversation.

        :returns: A copy of the conversation.
        """
        child = Chat()
        child.messages = deepcopy(self.messages)
        return child

    def after(self, parent: "Chat") -> "Chat":
        """
        Create a chat with only messages after the last common
        message (that appears in both parent conversation and
        this one).

        :param parent: Parent conversation.
        :returns: A new conversation with only new messages.
        """
        index = 0
        while (
            index < min(len(self.messages), len(parent.messages))
            and self.messages[index] == parent.messages[index]
        ):
            index += 1

        child = Chat()
        child.messages = [deepcopy(msg) for msg in self.messages[index:]]
        return child

    def last(self) -> dict[str, str] | None:
        """
        Get the last message in the conversation.

        :returns: The last message, or None if the conversation is empty.
        """
        return self.messages[-1] if self.messages else None

    def __iter__(self) -> Iterator[dict[str, str]]:
        """
        Iterate over the messages in the conversation.

        :returns: An iterator over the messages.
        """
        return iter(self.messages)

    def __repr__(self) -> str:
        return f"<Chat({self.messages})>"
