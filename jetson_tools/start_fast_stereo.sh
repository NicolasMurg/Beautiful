#!/usr/bin/env bash
set -euo pipefail
perception_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TORCHDYNAMO_DISABLE=1
exec "$perception_dir/.venv-ffs/bin/python" "$perception_dir/live_dual_yolo_depth.py" \
  --stereo-calibration "$perception_dir/stereo_calibration.npz" \
  --baseline-m 0.10795 --stereo-backend fast-foundation \
  --reading-age-ms 1000 --smoothing-window 1 "$@"
