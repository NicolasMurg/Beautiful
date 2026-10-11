#!/usr/bin/env python3
"""Play a fixed Piper coach voice test through the connected JBL."""
import argparse
from pathlib import Path
import tempfile
from voice_coach_test import ASSETS, Voice, find_sink, run, synthesize

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--text', default='Hello! I am your AI pool coach. Take your time, keep your stroke smooth, and aim carefully. This is a voice test through the JBL speaker.')
args = parser.parse_args()
voice = Voice('piper', ASSETS/'piper/piper', ASSETS/'en_US-lessac-medium.onnx')
sink = find_sink()
if voice.engine != 'piper' or sink is None:
    parser.exit(1, 'Piper voice or JBL output is unavailable.\n')
with tempfile.TemporaryDirectory(prefix='coach-voice-test-') as tmp:
    path = Path(tmp)/'voice.wav'
    synthesize(voice, args.text, path)
    run('pw-play', '--target', str(sink[2]), str(path), timeout=60)
print('Coach voice playback completed.')
