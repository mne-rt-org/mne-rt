"""Shared pytest fixtures and configuration."""
import sys
from pathlib import Path

# Ensure the package source is on sys.path for all tests
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
