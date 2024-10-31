from __future__ import annotations

import re
from base64 import b64encode
from enum import Enum
from mimetypes import guess_type
from typing import Any

from pydantic import BaseModel, field_validator

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
    """
    Part of an LLM chat message with a specific type:

    * `text`: Textual content
    * `image`: Image (PNG or JPG) as a data URL or HTTP/HTTPS URL
        (HTTP/HTTPS supported only by OpenAI)
    * `tool_call`: Tool call made by the assistant
    * `tool_response`: Tool response (provided by the client)

    Image can be provided as either a data URL, raw image data (bytes),
    or an HTTP(S) URL. If provided as raw image data in supported format
    (PNG or JPEG), it will be automatically converted to a data URL.
    """

    type: ContentType
    text: str | None = None
    image: str | None = None
    tool_call: ToolCall | None = None
    tool_response: ToolResponse | None = None

    @field_validator("image", mode="before")
    @classmethod
    def validate_image(cls, v):
        if not v:
            return None

        if isinstance(v, str):
            for scheme in ["data", "http", "https"]:
                if v.startswith(f"{scheme}:"):
                    return v
            raise ValueError("Image should be a data or HTTP(S) URL")

        if not isinstance(v, bytes):
            raise ValueError("Image should be a string (URL) or bytes (raw image data)")

        for typ, magic in MAGIC_BYTES.items():
            if v.startswith(magic):
                mime_type = typ
                break
        else:
            raise ValueError("Unsupported image format")

        return f"data:{mime_type};base64,{b64encode(v).decode('ascii')}"

    @property
    def image_data(self) -> str | None:
        """
        Return base64-encoded image data if possible.

        For images provided as HTTP(S) URLs, this will return None.

        :return: Base64-encoded image data or None
        """
        if not self.image:
            return None

        if self.image.startswith("data:"):
            return self.image.split(",", 1)[1]

        return None

    @property
    def image_mime_type(self) -> str | None:
        """
        Return the MIME type of the image if possible.

        :return: MIME type of the image or None
        """

        if not self.image:
            return None

        # If it's a data URL, extract the mime type
        m = DATA_URI_MIME_TYPE_PATTERN.match(self.image)
        if m:
            return m.group(1)

        # Otherwise, try to guess based off the URL
        if self.image.startswith("http:"):
            return guess_type(self.image)[0]

        return None


class Message(BaseModel):
    """
    A message in an LLM chat.
    """

    role: Role
    content: list[ContentPart] | None = None

    @classmethod
    def create(
        cls,
        role: Role,
        *,
        text: str | None = None,
        images: list[str | bytes] | None = None,
        tool_calls: list[ToolCall] | None = None,
        tool_responses: dict[str, str] | None = None,
    ) -> "Message":
        """
        Helper method to create a message with the given role.

        When providing tool responses, the dictionary should map tool call IDs
        to response content.

        :param role: Role of the message
        :param text: Text content, if any
        :param images: Image(s) attached to the message, if any
        :param tool_calls: Tool calls, if this is an assistant message
        :param tool_responses: Tool responses, if this is a tool response message
        :return: Message instance
        """
        content = []
        if text:
            content.append(ContentPart(type=ContentType.text, text=text))
        if images:
            for image in images:
                content.append(ContentPart(type=ContentType.image, image=image))
        if tool_calls:
            for call in tool_calls:
                content.append(ContentPart(type=ContentType.tool_call, tool_call=call))
        if tool_responses:
            for call_id, response in tool_responses.items():
                content.append(
                    ContentPart(
                        type=ContentType.tool_response,
                        tool_response=ToolResponse(
                            call=ToolCall(
                                id=call_id,
                                name="",
                                arguments={},
                            ),
                            response=response,
                        ),
                    )
                )
        return cls(role=role, content=content)

    @classmethod
    def system(cls, text: str, images: list[str | bytes] | None = None) -> "Message":
        """
        Creates a system message with the given text and optional images.

        :param text: The text content of the system message.
        :param images: Optional, a list of images associated with the message,
            which can be either strings or bytes.
        :return: A new Message instance.
        """
        return cls.create(Role.system, text=text, images=images)

    @classmethod
    def user(cls, text: str, images: list[str | bytes] | None = None) -> "Message":
        """
        Creates a user message with the given text and optional images.

        :param text: The text content of the message.
        :param images: Optional, a list of images associated with the message,
            which can be either strings or bytes.
        :return: A new Message instance.
        """
        return cls.create(Role.user, text=text, images=images)

    @classmethod
    def assistant(
        cls,
        text: str | None,
        tool_calls: list[ToolCall] | None = None,
    ) -> "Message":
        """
        Creates an assistant message with the specified text and tool calls.

        :param text: The text content of the assistant message (optional).
        :param tool_calls: Optional, list of tool calls that the assistant makes.
        :return: A new Message instance.
        """
        return cls.create(role=Role.assistant, text=text, tool_calls=tool_calls)

    @classmethod
    def tool(cls, tool_responses: dict[str, str]) -> "Message":
        """
        Creates a tool response message with the specified tool responses.

        :param tool_responses: Dictionary mapping tool call IDs to response content.
        :return: A new Message instance.
        """
        return cls.create(role=Role.tool, tool_responses=tool_responses)


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

    def system(self, text: str, images: list[str | bytes] | None = None):
        self.messages.append(Message.system(text, images))
        return self

    def user(self, text: str | None, images: list[str | bytes] | None = None):
        self.messages.append(Message.user(text, images))
        return self

    def assistant(self, text: str | None, tool_calls: list[ToolCall] | None = None):
        self.messages.append(Message.assistant(text, tool_calls))
        return self

    def tool(self, tool_responses: dict[str, str]):
        self.messages.append(Message.tool(tool_responses))
        return self

    def dump(self) -> list[dict[str, Any]]:
        return [m.model_dump(exclude_none=True) for m in self.messages]

    @classmethod
    def load(cls, data: list[dict[str, Any]]) -> "Chat":
        c = cls()
        c.messages = [Message(**m) for m in data]
        return c

    def clone(self) -> "Chat":
        c = Chat()
        c.messages = [m.model_copy(deep=True) for m in self.messages]
