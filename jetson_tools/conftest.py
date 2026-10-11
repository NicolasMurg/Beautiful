"""Hardware smoke scripts are invoked explicitly, never during test collection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
collect_ignore = ['test_fast_stereo.py', 'test_coach_voice.py']
