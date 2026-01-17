"""
Docstring Compliance Checker.

This script traverses Python files in specified directories or file paths,
parses them using the `ast` module, and checks for the presence of docstrings
in modules, classes, and functions.

It identifies missing docstrings to help maintain documentation standards
across the codebase.

Usage:
    python check_docstrings.py <path1> [path2 ...]
"""

import ast
import os
import sys


def check_path(path):
    """
    Check a single file for missing docstrings using AST parsing.

    Args:
        path (str): The file path to check.

    Returns:
        list: A list of strings describing any missing docstrings found.
              Returns an empty list if the file is not a Python file,
              has syntax errors, or is fully documented.
    """
    missing = []
    if os.path.isfile(path):
        if not path.endswith(".py"):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except SyntaxError:
            print(f"Syntax error in {path}")
            return []

        if not ast.get_docstring(tree):
            missing.append(f"{path}: (Module) Missing docstring")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_") and not node.name.startswith("__"):
                    continue

                if not ast.get_docstring(node):
                    kind = "Class" if isinstance(node, ast.ClassDef) else "Function"
                    missing.append(f"{path}: ({kind}) {node.name} Missing docstring")
    return missing


def check_docstrings_recursive(directory):
    """
    Recursively check a directory for Python files with missing docstrings.

    Args:
        directory (str): The root directory to search.

    Returns:
        list: A list of strings describing missing docstrings in all found Python files.
    """
    missing = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            missing.extend(check_path(os.path.join(root, file)))
    return missing


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_docstrings.py <path1> [path2 ...]")
        sys.exit(1)

    all_missing = []
    for arg in sys.argv[1:]:
        if os.path.isfile(arg):
            all_missing.extend(check_path(arg))
        elif os.path.isdir(arg):
            all_missing.extend(check_docstrings_recursive(arg))
        else:
            print(f"Skipping invalid path: {arg}")

    if all_missing:
        print("Missing Docstrings Found:")
        for item in all_missing:
            print(item)
    else:
        print("No missing docstrings found!")
