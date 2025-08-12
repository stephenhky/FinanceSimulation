#!/usr/bin/env python3
"""
Script to generate documentation for the finsim project using Sphinx.

This script automates the process of generating HTML documentation
from the source code docstrings using Sphinx.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Main function to generate documentation."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"
    
    # Change to the docs directory
    os.chdir(docs_dir)
    
    # Run sphinx-apidoc to generate .rst files from the source code
    print("Generating API documentation files...")
    subprocess.run([
        "sphinx-apidoc", 
        "-o", ".", 
        "--force", 
        "--no-toc", 
        "--module-first",
        str(project_root / "finsim")
    ], check=True)
    
    # Build the HTML documentation
    print("Building HTML documentation...")
    subprocess.run(["make", "html"], check=True)
    
    print(f"Documentation generated successfully!")
    print(f"Open {docs_dir / '_build' / 'html' / 'index.html'} in your browser to view the documentation.")


if __name__ == "__main__":
    main()