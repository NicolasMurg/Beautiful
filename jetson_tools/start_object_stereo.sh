#!/usr/bin/env bash
set -euo pipefail
perception_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$perception_dir/start_fast_stereo.sh" --model "$perception_dir/yoloe-26s-balls-only.pt" --imgsz 640 --ball-detail --billiards-only --auto-score --pocket-model "$perception_dir/training/pockets_refined/weights/best.pt" "$@"
