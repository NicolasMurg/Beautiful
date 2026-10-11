"""Write a printable SVG checkerboard; measure a printed square before calibrating."""

import argparse
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cols', type=int, default=9, help='Inner corners per row')
    parser.add_argument('--rows', type=int, default=6, help='Inner corners per column')
    parser.add_argument('--square-mm', type=float, default=18)
    parser.add_argument('--output', type=Path, default=Path('stereo_checkerboard.svg'))
    args = parser.parse_args()
    if min(args.cols, args.rows) < 3 or not math.isfinite(args.square_mm) or args.square_mm <= 0:
        parser.error('Use positive square size and at least 3 inner corners per axis')
    square = args.square_mm
    width = (args.cols + 1) * square + 20
    height = (args.rows + 1) * square + 30
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" '
             f'viewBox="0 0 {width} {height}">',
             f'<rect width="{width}" height="{height}" fill="white"/>']
    for row in range(args.rows + 1):
        for col in range(args.cols + 1):
            if (row + col) % 2 == 0:
                lines.append(f'<rect x="{10 + col * square}" y="{10 + row * square}" '
                             f'width="{square}" height="{square}" fill="black"/>')
    lines.extend([f'<text x="10" y="{height - 9}" font-size="3">'
                  f'{args.cols} x {args.rows} INNER corners; {square:g} mm squares</text>',
                  f'<text x="10" y="{height - 4}" font-size="3">'
                  'Print at 100%; measure a printed square before calibration.</text>', '</svg>'])
    args.output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}; print without fit-to-page and verify square size.')


if __name__ == '__main__':
    main()
