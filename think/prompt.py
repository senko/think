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
