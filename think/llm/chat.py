from __future__ import annotations

import binascii
import json
import re
from base64 import b64decode, b64encode
from enum import Enum
from mimetypes import guess_type
from typing import Any

from pydantic import BaseModel, field_validator

from .tool import ToolCall, ToolResponse

DATA_URI_MIME_TYPE_PATTERN = re.compile(r"data:([^;]+);base64")
IMAGE_MAGIC_BYTES = {
    "image/jpeg": b"\xff\xd8\xff\xe0",
    "image/png": b"\x89PNG\x0d\x0a\x1a\x0a",
}
DOCUMENT_MAGIC_BYTES = {
    "application/pdf": b"%PDF-",
}


class Role(str, Enum):
    """
    Message role (sender identity) in an LLM chat.
    """

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ContentType(str, Enum):
    """
    Content type of a part of an LLM chat message.
    """

    text = "text"
    image = "image"
    document = "document"
    tool_call = "tool_call"
    tool_response = "tool_response"


def _validate_file(
    value: Any,
    type_desc: str,
    magic_bytes: dict[str, bytes],
) -> str | None:
    """Generic file validator/converter for image/document fields."""

    if not value:
        return None

    if isinstance(value, str):
        for scheme in ["data", "http", "https"]:
            if value.startswith(f"{scheme}:"):
                return value

        try:
            value = b64decode(value)
        except (ValueError, binascii.Error):
            raise ValueError(
                f"{type_desc.title()} should be data:, http: or https: URL or base64-encoded raw {type_desc} data"
            )

    if not isinstance(value, bytes):
        raise ValueError(
            f"{type_desc.title()} should be a string (URL) or bytes (raw {type_desc} data)"
        )

    for typ, magic in magic_bytes.items():
        if value.startswith(magic):
            mime_type = typ
            break
    else:
        raise ValueError(f"Unsupported {type_desc} format")

    return f"data:{mime_type};base64,{b64encode(value).decode('ascii')}"


def image_url(value: Any) -> str | None:
    """
    Converts raw image data to a data URL.

    :param value: The raw image data or URL to be converted.
    :return: A data URL representing the image.
    """
    return _validate_file(value, "image", IMAGE_MAGIC_BYTES)


def document_url(value: Any) -> str | None:
    """
    Converts raw document data to a data URL.

    :param value: The raw document data or URL to be converted.
    :return: A data URL representing the document.
    """
    return _validate_file(value, "document", DOCUMENT_MAGIC_BYTES)


def _get_file_b64(data: str) -> str | None:
    """
    Return base64-encoded file data if possible.

    For files provided as HTTP(S) URLs, this will return None.

    :return: Base64-encoded file data or None
    """
    if not data:
        return None

    if data.startswith("data:"):
        return data.split(",", 1)[1]

    return None


def _get_file_mime_type(data: str) -> str | None:
    """
    Return the MIME type of the file if possible.

    :return: MIME type of the file or None
    """

    if not data:
        return None

    # If it's a data URL, extract the mime type
    m = DATA_URI_MIME_TYPE_PATTERN.match(data)
    if m:
        return m.group(1)

    # Otherwise, try to guess based off the URL
    if data.startswith("http:"):
        return guess_type(data)[0]

    return None


class ContentPart(BaseModel):
    """
    Part of an LLM chat message with a specific type:

    * `text`: Textual content
    * `image`: Image (PNG or JPG) as a data URL or HTTP/HTTPS URL
        (HTTP/HTTPS supported only by OpenAI)
    * `document`: Document in PDF format, as a data URL or HTTP/HTTPS URL
        (HTTP/HTTPS supported only by OpenAI)
    * `tool_call`: Tool call made by the assistant
    * `tool_response`: Tool response (provided by the client)

    Image/document can be provided as either a data URL, raw data (bytes),
    or an HTTP(S) URL. If provided as raw data in supported format
    (PNG or JPEG for images, PDF for documents), it will be
    automatically converted to a data URL.

    Note: not all content types are supported by all AI models.
    """

    type: ContentType
    text: str | None = None
    image: str | None = None
    document: str | None = None
    tool_call: ToolCall | None = None
    tool_response: ToolResponse | None = None

    @field_validator("image", mode="before")
    @classmethod
    def validate_image(cls, v):
        """Pydantic validator/converter for the image field."""
        return image_url(v)

    @property
    def is_image_url(self) -> bool:
        """
        Return True if the image is an HTTP(S) URL.

        :return: True if the image is an HTTP(S) URL
        """
        return self.image and self.image.startswith(("http:", "https:"))

    @property
    def image_data(self) -> str | None:
        """
        Return base64-encoded image data if possible.

        For images provided as HTTP(S) URLs, this will return None.

        :return: Base64-encoded image data or None
        """
        return _get_file_b64(self.image)

    @property
    def image_bytes(self) -> bytes | None:
        """
        Return raw image data if possible.

        For images provided as data URLs, this will return None.

        :return: Raw image data (as byte string), or None
        """
        encoded = self.image_data
        return b64decode(encoded) if encoded else None

    @property
    def image_mime_type(self) -> str | None:
        """
        Return the MIME type of the image if possible.

        :return: MIME type of the image or None
        """
        return _get_file_mime_type(self.image)

    @field_validator("document", mode="before")
    @classmethod
    def validate_document(cls, v):
        """Pydantic validator/converter for the document field."""
        return document_url(v)

    @property
    def is_document_url(self) -> bool:
        """
        Return True if the document is an HTTP(S) URL.

        :return: True if the document is an HTTP(S) URL
        """
        return self.document and self.document.startswith(("http:", "https:"))

    @property
    def document_data(self) -> str | None:
        """
        Return base64-encoded document data if possible.

        For documents provided as HTTP(S) URLs, this will return None.

        :return: Base64-encoded document data or None
        """
        return _get_file_b64(self.document)

    @property
    def document_bytes(self) -> bytes | None:
        """
        Return raw document data if possible.

        For documents provided as HTTP(S) URLs, this will return None.

        :return: Raw document data (as byte string), or None
        """
        encoded = self.document_data
        return b64decode(encoded) if encoded else None

    @property
    def document_mime_type(self) -> str | None:
        """
        Return the MIME type of the document if possible.

        :return: MIME type of the document or None
        """
        return _get_file_mime_type(self.document)


class Message(BaseModel):
    """
    A message in an LLM chat.

    Provider-independent representation of a message,
    to be converted from/to provider-specific format
    by the appropriate adapter.

    Note: not all roles are supported by all AI models.

    If the LLM call specified a parser and the AI reply
    was successfully parsed, the `parsed` field will contain
    the parsed output, otherwise it will be None.
    """

    role: Role
    content: list[ContentPart]
    parsed: Any | None = None

    @classmethod
    def create(
        cls,
        role: Role,
        *,
        text: str | None = None,
        images: list[str | bytes] | None = None,
        documents: list[str | bytes] | None = None,
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
        :param documents: Document(s) attached to the message, if any
        :param tool_calls: Tool calls, if this is an assistant message
        :param tool_responses: Tool responses, if this is a tool response
            message.
        :return: Message instance
        """
        content = []
        if text:
            content.append(ContentPart(type=ContentType.text, text=text))
        if images:
            for image in images:
                content.append(ContentPart(type=ContentType.image, image=image))
        if documents:
            for document in documents:
                content.append(
                    ContentPart(
                        type=ContentType.document,
                        document=document,
                    )
                )
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
    def system(
        cls,
        text: str,
        images: list[str | bytes] | None = None,
        documents: list[str | bytes] | None = None,
    ) -> "Message":
        """
        Creates a system message with the given text and optional images.

        :param text: The text content of the system message.
        :param images: Optional, a list of images associated with the message,
            which can be either strings or bytes.
        :param documents: Optional, a list of documents associated with the message,
            which can be either strings or bytes.
        :return: A new Message instance.
        """
        return cls.create(
            Role.system,
            text=text,
            images=images,
            documents=documents,
        )

    @classmethod
    def user(
        cls,
        text: str | None,
        images: list[str | bytes] | None = None,
        documents: list[str | bytes] | None = None,
    ) -> "Message":
        """
        Creates a user message with the given text and optional images.

        :param text: The text content of the message.
        :param images: Optional, a list of images associated with the message,
            which can be either strings or bytes.
        :param documents: Optional, a list of documents associated with the message,
            which can be either strings or bytes.
        :return: A new Message instance.
        """
        return cls.create(Role.user, text=text, images=images, documents=documents)

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
    """
    A conversation with an LLM.

    Provider-independent representation of messages exchanged with
    the LLM, to be converted from/to provider-specific format
    by the appropriate adapter.
    """

    messages: list[Message]

    def __init__(self, system_message: str | None = None):
        """
        Initialize a new chat with an optional system message.

        :param system_message: Optional system message to include
            in the chat.
        """
        self.messages = []
        if system_message:
            self.messages.append(Message.system(system_message))

    def __iter__(self):
        """Iterate over the messages in the chat."""
        return iter(self.messages)

    def __len__(self):
        """Return the number of messages in the chat."""
        return len(self.messages)

    def __str__(self) -> str:
        """Return a JSON representation of the chat."""
        return json.dumps(self.dump())

    def system(
        self,
        text: str,
        images: list[str | bytes] | None = None,
        documents: list[str | bytes] | None = None,
    ) -> "Chat":
        """
        Add a system message to the chat.

        :param text: The text content of the system message.
        :param images: Optional, a list of images associated with
            the message, which can be either strings or bytes.
        :param documents: Optional, a list of documents associated with
            the message, which can be either strings or bytes.
        :return: The chat instance, for chaining.
        """
        self.messages.append(Message.system(text, images, documents))
        return self

    def user(
        self,
        text: str | None,
        images: list[str | bytes] | None = None,
        documents: list[str | bytes] | None = None,
    ) -> "Chat":
        """
        Add a user message to the chat.

        :param text: The text content of the system message.
        :param images: Optional, a list of images associated with
            the message, which can be either strings or bytes.
        :param documents: Optional, a list of documents associated with
            the message, which can be either strings or bytes.
        :return: The chat instance, for chaining.
        """
        self.messages.append(Message.user(text, images, documents))
        return self

    def assistant(
        self, text: str | None, tool_calls: list[ToolCall] | None = None
    ) -> "Chat":
        """
        Add an assistant message to the chat.

        :param text: The text content of the assistant message.
        :param tool_calls: Optional, list of tool calls that the assistant makes.
        :return: The chat instance, for chaining.
        """
        self.messages.append(Message.assistant(text, tool_calls))
        return self

    def tool(self, tool_responses: dict[str, str]) -> "Chat":
        """
        Add a tool response message to the chat.

        :param tool_responses: Dictionary mapping tool call IDs to response content.
        :return: The chat instance, for chaining.
        """
        self.messages.append(Message.tool(tool_responses))
        return self

    def dump(self) -> list[dict[str, Any]]:
        """
        Dump the chat to a JSON-serializable format.

        Note that the format is provider-independent. It's useful for
        storing and loading chats, but must be converted to the
        provider-specific format by the appropriate adapter.

        :return: JSON-serializable representation of the chat.
        """
        return [m.model_dump(exclude_none=True) for m in self.messages]

    @classmethod
    def load(cls, data: list[dict[str, Any]]) -> "Chat":
        """
        Load a chat from a JSON-serializable format.

        Loads a chat saved using the `dump` method.

        :param data: JSON-serializable representation of the chat.
        :return: Chat instance.
        """
        c = cls()
        c.messages = [Message.model_validate(m) for m in data]
        return c

    def clone(self) -> "Chat":
        """
        Return a copy of the chat.

        Performs a deep-copy, so there's no shared state between the
        original and the cloned chat.

        :return: A copy of the chat.
        """
        c = Chat()
        c.messages = [m.model_copy(deep=True) for m in self.messages]
        return c
