#!/usr/bin/env bash
set -euo pipefail
perception_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$perception_dir/start_fast_stereo.sh" --model "$perception_dir/yoloe-26s-objects.pt" --imgsz 640 "$@"
