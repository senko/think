#!/usr/bin/env python3
"""
Build script for generating Think documentation.

This script extracts docstrings from Python modules and renders
them into a comprehensive Markdown documentation file using a
Jinja2 template.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jinja2

# Define mappings of topics to files
MODULE_ROOT = Path(__file__).parent.parent / "think"
TOPICS = {
    "Basic LLM Use": [
        (MODULE_ROOT / "llm" / "base.py", "Core LLM Functionality"),
        (MODULE_ROOT / "ai.py", "High-level API"),
    ],
    "Supported Providers": [
        (MODULE_ROOT / "llm" / "__init__.py", "Overview"),
        (MODULE_ROOT / "llm" / "openai.py", "OpenAI"),
        (MODULE_ROOT / "llm" / "anthropic.py", "Anthropic"),
        (MODULE_ROOT / "llm" / "google.py", "Google (Gemini)"),
        (MODULE_ROOT / "llm" / "bedrock.py", "Amazon (Bedrock)"),
        (MODULE_ROOT / "llm" / "groq.py", "Groq"),
        (MODULE_ROOT / "llm" / "ollama.py", "Ollama"),
    ],
    "Chat/Conversation Manipulation": [
        (MODULE_ROOT / "llm" / "chat.py", "Chat Functionality")
    ],
    "Prompting": [(MODULE_ROOT / "prompt.py", "Prompt Templates")],
    "Structured Outputs and Parsing": [
        (MODULE_ROOT / "parser.py", "Parsing Functionality")
    ],
    "Vision and Document Handling": [
        (MODULE_ROOT / "llm" / "chat.py", "Vision Capabilities"),
    ],
    "Streaming": [(MODULE_ROOT / "llm" / "base.py", "Streaming Responses")],
    "Tool Use": [(MODULE_ROOT / "llm" / "tool.py", "Tool Integration")],
    "RAG (Retrieval-Augmented Generation)": [
        (MODULE_ROOT / "rag" / "base.py", "RAG Base Functionality"),
        (MODULE_ROOT / "rag" / "chroma_rag.py", "ChromaDB Integration"),
        (MODULE_ROOT / "rag" / "pinecone_rag.py", "Pinecone Integration"),
        (MODULE_ROOT / "rag" / "txtai_rag.py", "TxtAI Integration"),
        (MODULE_ROOT / "rag" / "eval.py", "RAG Evaluation"),
    ],
    "Agents": [(MODULE_ROOT / "agent.py", "Building Agents")],
}


def extract_module_docstring(file_path: Path) -> Optional[str]:
    """Extract the module-level docstring from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            module_ast = ast.parse(f.read())
        if (
            module_ast.body
            and isinstance(module_ast.body[0], ast.Expr)
            and isinstance(module_ast.body[0].value, ast.Constant)
        ):
            return module_ast.body[0].value.value
        return None
    except Exception as e:
        print(f"Error extracting docstring from {file_path}: {e}")
        return None


def get_function_signature(
    func_node: ast.FunctionDef, class_name: Optional[str] = None
) -> str:
    """Generate a readable function signature."""
    args = []

    # Handle arguments
    for arg in func_node.args.args:
        if arg.annotation:
            annotation = ast.unparse(arg.annotation).strip()
            args.append(f"{arg.arg}: {annotation}")
        else:
            args.append(arg.arg)

    # Handle *args
    if func_node.args.vararg:
        if func_node.args.vararg.annotation:
            annotation = ast.unparse(func_node.args.vararg.annotation).strip()
            args.append(f"*{func_node.args.vararg.arg}: {annotation}")
        else:
            args.append(f"*{func_node.args.vararg.arg}")

    # Handle **kwargs
    if func_node.args.kwarg:
        if func_node.args.kwarg.annotation:
            annotation = ast.unparse(func_node.args.kwarg.annotation).strip()
            args.append(f"**{func_node.args.kwarg.arg}: {annotation}")
        else:
            args.append(f"**{func_node.args.kwarg.arg}")

    # Handle return type
    returns = ""
    if func_node.returns:
        returns = f" -> {ast.unparse(func_node.returns).strip()}"

    if class_name:
        prefix = f"{class_name}."
    else:
        prefix = ""

    return f"{prefix}{func_node.name}({', '.join(args)}){returns}"


def extract_class_info(node: ast.ClassDef) -> Dict:
    """Extract information about a class from its AST node."""
    class_info = {
        "name": node.name,
        "description": ast.get_docstring(node) or "",
        "methods": [],
    }

    for item in node.body:
        if isinstance(item, ast.FunctionDef) and (
            not item.name.startswith("_") or item.name == "__init__"
        ):
            method_info = {
                "name": item.name,
                "signature": get_function_signature(item, node.name),
                "description": ast.get_docstring(item) or "",
            }
            class_info["methods"].append(method_info)

    return class_info


def extract_function_info(node: ast.FunctionDef) -> Dict:
    """Extract information about a function from its AST node."""
    return {
        "name": node.name,
        "signature": get_function_signature(node),
        "description": ast.get_docstring(node) or "",
    }


def process_module(file_path: Path, module_name: str = None) -> Dict:
    """Process a Python module to extract docstrings and API information."""
    with open(file_path, "r", encoding="utf-8") as f:
        module_ast = ast.parse(f.read())

    if not module_name:
        module_name = file_path.stem

    module_info = {
        "name": module_name,
        "description": extract_module_docstring(file_path) or "",
        "classes": [],
        "functions": [],
    }

    for node in module_ast.body:
        # Only include public classes and functions (not starting with _)
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            module_info["classes"].append(extract_class_info(node))
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            module_info["functions"].append(extract_function_info(node))

    return module_info


def collect_guide_sections(think_dir: Path) -> List[Dict]:
    """Collect guide sections from module docstrings."""
    guide_sections = []

    for title, files in TOPICS.items():
        section_content = []

        for file_info in files:
            file_path, subtopic = file_info
            if file_path.exists():
                docstring = extract_module_docstring(file_path)
                if docstring:
                    if (
                        len(files) > 1
                    ):  # If there are multiple files for this topic, add subtopic headers
                        section_content.append(f"#### {subtopic}\n\n{docstring}")
                    else:
                        section_content.append(docstring)

        if section_content:
            guide_sections.append(
                {"title": title, "content": "\n\n".join(section_content)}
            )

    return guide_sections


def collect_api_reference(think_dir: Path) -> List[Dict]:
    """Collect API reference from class/method/function docstrings."""
    api_modules = []

    module_paths = set()

    # Get all Python files in the project
    for file_path in think_dir.glob("**/*.py"):
        # Skip experimental files and test files
        if "hf.py" in str(file_path) or "/tests/" in str(file_path):
            continue
        module_paths.add(file_path)

    # Process files in sorted order for consistent output
    for file_path in sorted(module_paths):
        rel_path = file_path.relative_to(think_dir)
        parts = list(rel_path.parts)

        # Create module name
        if len(parts) > 1:
            module_name = f"think.{'.'.join(p.replace('.py', '') for p in parts)}"
        else:
            module_name = f"think.{parts[0].replace('.py', '')}"

        module_info = process_module(file_path, module_name)
        if module_info["classes"] or module_info["functions"]:
            api_modules.append(module_info)

    return api_modules


def extract_readme_sections(readme_path: Path) -> Tuple[str, str]:
    """Extract the introduction and quickstart sections from README.md."""
    readme_text = readme_path.read_text(encoding="utf-8")

    # Extract the introduction (everything before the first ## heading)
    intro_match = re.search(r"^# Think\n\n(.*?)(?=\n## )", readme_text, re.DOTALL)
    intro = intro_match.group(1).strip() if intro_match else ""

    # Extract the quickstart section
    quickstart_match = re.search(
        r"## Quickstart\n\n(.*?)(?=\n## )", readme_text, re.DOTALL
    )
    quickstart = quickstart_match.group(1).strip() if quickstart_match else ""

    return intro, quickstart


def extract_examples_from_readme(readme_path: Path) -> Dict[str, str]:
    """Extract code examples from README.md."""
    readme_text = readme_path.read_text(encoding="utf-8")

    # Find all Python code blocks with example comments
    example_pattern = r"```python\n# example: ([a-zA-Z0-9_]+\.py)\n(.*?)```"
    examples = {}

    for match in re.finditer(example_pattern, readme_text, re.DOTALL):
        filename, code = match.groups()
        examples[filename] = code.strip()

    return examples


def render_template(template_path: Path, output_path: Path, context: Dict) -> None:
    """Render the Jinja2 template with the provided context."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_path.parent),
        autoescape=False,  # We want raw markdown output
    )
    template = env.get_template(template_path.name)
    rendered = template.render(**context)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


def main():
    """Build the documentation."""
    # Define paths
    repo_dir = Path(__file__).parent.parent
    think_dir = repo_dir / "think"
    template_path = repo_dir / "docs" / "docs.md.jinja2"
    output_path = repo_dir / "docs" / "docs.md"
    readme_path = repo_dir / "README.md"

    intro, quickstart = extract_readme_sections(readme_path)
    guide_sections = collect_guide_sections(think_dir)
    api_modules = collect_api_reference(think_dir)
    context = {
        "intro": intro,
        "quickstart": quickstart,
        "guide_sections": guide_sections,
        "api_modules": api_modules,
    }
    render_template(template_path, output_path, context)

    print(f"Documentation built successfully and saved to {output_path}")


if __name__ == "__main__":
    main()
