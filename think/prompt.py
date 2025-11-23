"""
# Prompt Templates

The `prompt` module provides functionality for creating and managing templates
for prompts to be sent to LLMs. It's built on top of Jinja2 and offers both
string-based and file-based templating.

## String Templates

The simplest way to use templates is with `JinjaStringTemplate`:

```python
# example: string_template.py
from think.prompt import JinjaStringTemplate

template = JinjaStringTemplate()
prompt = template("Hello, my name is {{ name }} and I'm {{ age }} years old.",
                  name="Alice", age=30)
print(prompt)  # Outputs: Hello, my name is Alice and I'm 30 years old.
```

## File Templates

For more complex prompts, you can use file-based templates:

```python
# example: file_template.py
from pathlib import Path
from think.prompt import JinjaFileTemplate

# Create a template file
template_path = Path("my_template.txt")
template_path.write_text("Hello, my name is {{ name }} and I'm {{ age }} years old.")

# Use the template
template = JinjaFileTemplate(template_path.parent)
prompt = template("my_template.txt", name="Bob", age=25)
print(prompt)  # Outputs: Hello, my name is Bob and I'm 25 years old.
```

## Multi-line Templates

When working with multi-line templates, the `strip_block` function helps
preserve the relative indentation while removing the overall indentation:

```python
# example: multiline_template.py
from think.prompt import JinjaStringTemplate, strip_block

template = JinjaStringTemplate()
prompt_text = strip_block('''
    System:
        You are a helpful assistant.

    User:
        {{ question }}
''')

prompt = template(prompt_text, question="How does photosynthesis work?")
print(prompt)
```

## Using with LLMs

Templates integrate seamlessly with the Think LLM interface:

```python
# example: template_with_llm.py
import asyncio
from think import LLM, ask
from think.prompt import JinjaStringTemplate

llm = LLM.from_url("openai:///gpt-3.5-turbo")
template = JinjaStringTemplate()

async def main():
    prompt = template("Write a {{ length }} poem about {{ topic }}.",
                     length="short", topic="artificial intelligence")
    response = await ask(llm, prompt)
    print(response)

asyncio.run(main())
```

See also:
- [Basic LLM Use](#basic-llm-use) for using templates with LLMs
- [Structured Outputs and Parsing](#structured-outputs-and-parsing) for combining templates with structured outputs
"""

from pathlib import Path
from typing import Any, Optional

from jinja2 import BaseLoader, Environment, FileSystemLoader, StrictUndefined


def strip_block(txt: str) -> str:
    """
    Strip a multiline block

    Strips whitespace from each line in the block so that the indentation
    (if any) within the block is preserved, but the block itself is not
    indented. Also strips any trailing whitespace.

    :param txt: The block of text to strip.
    :return: The stripped block of text.
    """
    lines = txt.splitlines()
    min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    return "\n".join(line[min_indent:].rstrip() for line in lines).strip("\n")


class FormatTemplate:
    """
    Template renderer using str.format

    Instances of this class, when called with a template string
    and keyword arguments, will render and return the template.

    :param template: The template string to render.
    :param kwargs: Keyword arguments to pass to str.format.
    :return: The rendered template string.
    """

    def __call__(self, template: str, **kwargs: Any) -> str:
        """
        Render a template using str.format.

        :param template: The template string to render
        :param kwargs: Keyword arguments to substitute in the template
        :return: The rendered template string
        """
        return strip_block(template).format(**kwargs)


class BaseJinjaTemplate:
    """Base class for Jinja2 template renderers."""

    def __init__(self, loader: Optional[BaseLoader]):
        """
        Initialize the Jinja2 template environment.

        :param loader: Optional Jinja2 loader for template loading
        """
        self.env = Environment(
            loader=loader,
            autoescape=False,
            lstrip_blocks=True,
            trim_blocks=True,
            keep_trailing_newline=False,
            undefined=StrictUndefined,
        )


class JinjaStringTemplate(BaseJinjaTemplate):
    """
    String template renderer using Jinja2

    Instances of this class, when called with a template string
    and keyword arguments, will render and return the template.

    :param template: The template string to render.
    :param kwargs: Keyword arguments to pass to the template.
    :return: The rendered template string.
    """

    def __init__(self):
        """
        Initialize the string template renderer with no loader.
        """
        super().__init__(None)

    def __call__(self, template: str, **kwargs: Any) -> str:
        """
        Render a Jinja2 template from string.

        :param template: The template string to render
        :param kwargs: Keyword arguments to pass to the template
        :return: The rendered template string
        """
        tpl = self.env.from_string(strip_block(template))
        return tpl.render(**kwargs)


class JinjaFileTemplate(BaseJinjaTemplate):
    """
    File template renderer using Jinja2

    Instances of this class, when called with a template filename
    and keyword arguments, will render and return the template.

    Since this class uses the FileSystemLoader, the template
    may reference other templates using the Jinja2 include or
    extends statements.

    :param template: The template filename to render.
    :param kwargs: Keyword arguments to pass to the template.
    :return: The rendered template string.
    """

    def __init__(self, template_dir: str):
        """
        Initialize the file template renderer with a template directory.

        :param template_dir: Path to the directory containing template files
        :raises ValueError: If the template directory doesn't exist
        """
        if not Path(template_dir).is_dir():
            raise ValueError(f"Template directory does not exist: {template_dir}")
        super().__init__(FileSystemLoader(template_dir))

    def __call__(self, template: str, **kwargs: Any) -> str:
        """
        Render a Jinja2 template from file.

        :param template: The template filename to render
        :param kwargs: Keyword arguments to pass to the template
        :return: The rendered template string
        """
        tpl = self.env.get_template(template)
        return tpl.render(**kwargs)
