"""Resolve source checkout and local runtime without a machine-specific username."""
import os
from pathlib import Path

source_root = Path(__file__).resolve().parent.parent
default_project = source_root if (source_root / 'billiards.py').is_file() else Path.home() / 'Downloads/Beautiful'
PROJECT_ROOT = Path(os.environ.get('BEAUTIFUL_REPO', str(default_project))).expanduser()
RUNTIME_ROOT = Path(os.environ.get('JETSON_RUNTIME_ROOT', str(Path.home() / 'robot-perception'))).expanduser()
