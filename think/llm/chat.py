import re
from base64 import b64decode, b64encode
from enum import Enum
from mimetypes import guess_type
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from .tool import ToolCall, ToolResponse

DATA_URI_MIME_TYPE_PATTERN = re.compile(r"data:([^;]+);base64")
MAGIC_BYTES = {
    "image/jpeg": b"\xff\xd8\xff\xe0",
    "image/png": b"\x89PNG\x0d\x0a\x1a\x0a",
}


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ContentType(str, Enum):
    text = "text"
    image = "image"
    tool_call = "tool_call"
    tool_response = "tool_response"


class ContentPart(BaseModel):
    type: ContentType
    text: str | None = None
    image: str | bytes | None = None
    tool_call: ToolCall | None = None
    tool_response: ToolResponse | None = None

    def get_image_data(self) -> bytes | None:
        if not self.image:
            return None

        if isinstance(self.image, bytes):
            return self.image

        if isinstance(self.image, str):
            if self.image.startswith("data:"):
                try:
                    return b64decode(self.image.split(",")[1])
                except Exception:
                    return None

        return None

    def get_image_mime_type(self) -> str | None:
        if not self.image:
            return None

        if isinstance(self.image, bytes):
            # Try to detect image type based off the first few magic bytes
            for mime_type, magic_bytes in MAGIC_BYTES.items():
                if self.image.startswith(magic_bytes):
                    return mime_type

        if isinstance(self.image, str):
            # If it's a data URL, extract the mime type
            m = DATA_URI_MIME_TYPE_PATTERN.match(self.image)
            if m:
                return m.group(1)

            # Otherwise, try to guess based off the URL
            if self.image.startswith("http:"):
                return guess_type(self.image)[0]

        # FIXME log.warning("Could not determine mime type for image")
        return None


class Message(BaseModel):
    role: Role
    content: list[ContentPart] | None = None

    @classmethod
    def create(
        cls,
        role: Role,
        *,
        content: list[ContentPart] | None = None,
        text: str | None = None,
        images: list[str | bytes] | None = None,
        tool_calls: dict[str, Any] | None = None,
    ) -> "Message":
        parts = content[:] if content else []
        if text:
            parts.append(ContentPart(type="text", text=text))
        if images:
            for image in images:
                parts.append(ContentPart(type="image", image=image))
        if tool_calls:
            ...  # FIXME
        return cls(role=role, content=parts)

    @classmethod
    def system(cls, text: str, images: list[str | bytes] | None = None) -> "Message":
        return cls.create(Role.system, text=text, images=images)

    @classmethod
    def user(cls, text: str, images: list[str | bytes] | None = None) -> "Message":
        return cls.create(Role.user, text=text, images=images)

    @classmethod
    def assistant(
        cls,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> "Message":
        return cls(role=Role.assistant, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, call_id: str, response: str) -> "Message":
        return cls(role=Role.tool, tool_call_id=call_id, content=response)


class Chat:
    messages: list[Message]

    def __init__(self, system_message: str | None = None):
        self.messages = []
        if system_message:
            self.messages.append(Message.system(system_message))

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)

    def __str__(self) -> str:
        import json

        return json.dumps(self.dump())

    def system(self, content: str, image: str | bytes | None = None):
        self.messages.append(Message.system(content, image=image))
        return self

    def user(
        self, content: str | list[ContentPart] | None, image: str | bytes | None = None
    ):
        self.messages.append(Message.user(content, images=[image] if image else None))
        return self

    def assistant(
        self,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
    ):
        self.messages.append(Message.assistant(content, tool_calls=tool_calls))
        return self

    def tool(self, call_id: str, response: str):
        self.messages.append(Message.tool(call_id, response))
        return self

    def dump(self) -> list[dict[str, Any]]:
        # Note: OpenAI hates extra null fields and exclude_none config on
        # ContentPart or fields is ignored, so we have to use it explicitly.
        return [m.model_dump(exclude_none=True) for m in self.messages]

    @classmethod
    def load(cls, data: list[dict[str, Any]]) -> "Chat":
        c = cls()
        c.messages = [Message(**m) for m in data]
        return c

    def clone(self) -> "Chat":
        return self.load(self.dump())

    def extract_system(self) -> str:
        system_message = "\n\n".join(
            msg.content for msg in self if msg.role == Role.system
        )
        self.messages = [msg for msg in self if msg.role != Role.system]
        return system_message
