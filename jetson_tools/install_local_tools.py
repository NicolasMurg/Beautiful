#!/usr/bin/env python3
"""Copy source utilities into the existing Jetson runtime, backing up changed files."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=Path.home() / 'robot-perception')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    target = args.runtime_root.expanduser().resolve()
    if source == target:
        parser.error('Run this installer from the Git checkout, not the runtime directory')
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = target / 'source-backups' / stamp
    for path in sorted(source.iterdir()):
        if path.suffix not in ('.py', '.sh') and path.name != 'torch-constraints.txt':
            continue
        if path.name in ('conftest.py', 'install_local_tools.py'):
            continue
        output = target / path.name
        if output.exists():
            if output.read_bytes() == path.read_bytes():
                continue
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output, backup / path.name)
        shutil.copy2(path, output)
        print(output.name)
    print(f'Installed source into {target}; existing runtimes, models and calibration files preserved.')


if __name__ == '__main__':
    main()
