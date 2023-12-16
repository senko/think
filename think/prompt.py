from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, BaseLoader, FileSystemLoader, StrictUndefined


class FormatTemplate:
    def __call__(self, template: str, **kwargs: dict[str, Any]) -> str:
        return template.format(**kwargs)


class BaseJinjaTemplate:
    def __init__(self, loader: Optional[BaseLoader]):
        self.env = Environment(
            loader=loader,
            autoescape=False,
            lstrip_blocks=True,
            trim_blocks=True,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
        )


class JinjaStringTemplate(BaseJinjaTemplate):
    def __init__(self):
        super().__init__(None)

    def __call__(self, template: str, **kwargs: dict[str, Any]) -> str:
        tpl = self.env.from_string(template)
        return tpl.render(**kwargs)


class JinjaFileTemplate(BaseJinjaTemplate):
    def __init__(self, template_dir: str):
        if not Path(template_dir).is_dir():
            raise ValueError(f"Template directory does not exist: {template_dir}")
        super().__init__(FileSystemLoader(template_dir))

    def __call__(self, template: str, **kwargs: dict[str, Any]) -> str:
        tpl = self.env.get_template(template)
        return tpl.render(**kwargs)
