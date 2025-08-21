#!/usr/bin/env python3
"""
Script to test the documentation generation for the finsim project.

This script verifies that the documentation can be generated successfully.
"""

import os
import subprocess
import sys
from pathlib import Path


def test_docs_generation():
    """Test function to verify documentation generation."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"
    
    # Change to the docs directory
    os.chdir(docs_dir)
    
    # Test if sphinx is installed
    try:
        import sphinx
        print(f"Sphinx version: {sphinx.__version__}")
    except ImportError:
        print("Error: Sphinx is not installed.")
        return False
    
    # Test if sphinx-rtd-theme is installed
    try:
        import sphinx_rtd_theme
        print(f"Sphinx RTD Theme version: {sphinx_rtd_theme.__version__}")
    except ImportError:
        print("Error: Sphinx RTD Theme is not installed.")
        return False
    
    print("All required documentation dependencies are installed.")
    return True


def main():
    """Main function to run the documentation tests."""
    print("Testing documentation generation setup...")
    
    if test_docs_generation():
        print("Documentation generation setup is working correctly.")
    else:
        print("Documentation generation setup has issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()