#!/usr/bin/env python
#
# Python script to extract examples from README.md and store them in
# `examples` directory.

import os
import re
import pathlib
from typing import Dict, List


def extract_code_blocks(markdown_content: str) -> List[str]:
    """
    Extract Python code blocks from markdown content.

    Args:
        markdown_content: Markdown text containing code blocks

    Returns:
        List of extracted code blocks
    """
    # Pattern to match python code blocks: ```python ... ```
    pattern = r"```python\n(.*?)```"

    # Find all matches using re.DOTALL to match across multiple lines
    matches = re.findall(pattern, markdown_content, re.DOTALL)

    return matches


def parse_example_files(code_blocks: List[str]) -> Dict[str, str]:
    """
    Parse code blocks to find examples with filename comments.

    Args:
        code_blocks: List of code blocks extracted from markdown

    Returns:
        Dictionary mapping filenames to code content
    """
    examples = {}

    for block in code_blocks:
        match = re.search(r"# example: ([a-zA-Z0-9_\-]+\.py)", block)
        if match:
            filename = match.group(1)
            examples[filename] = block.strip() + "\n"

    return examples


def save_examples(examples: Dict[str, str], output_dir: pathlib.Path) -> None:
    """
    Save examples to files in the output directory.

    Args:
        examples: Dictionary mapping filenames to code content
        output_dir: Directory where examples will be saved
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename, content in examples.items():
        file_path = output_dir / filename
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Saved example: {file_path}")


def main():
    examples_dir = pathlib.Path(__file__).parent
    readme_path = examples_dir.parent / "README.md"

    try:
        with open(readme_path, "r") as f:
            readme_content = f.read()
    except FileNotFoundError:
        print(f"Error: README.md not found at {readme_path}")
        return

    code_blocks = extract_code_blocks(readme_content)
    examples = parse_example_files(code_blocks)

    if examples:
        save_examples(examples, examples_dir)
        print(f"Extracted {len(examples)} examples to {examples_dir}")
    else:
        print("No examples found in README.md")


if __name__ == "__main__":
    main()
