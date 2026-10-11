#!/usr/bin/env bash
set -euo pipefail
exec bash "${JETSON_RUNTIME_ROOT:-$HOME/robot-perception}"/recalibrate_table_cameras.sh \
  --max-vertical-error-px 1 \
  --holdout-views 5 --max-board-error-cm 1 \
  --output "${JETSON_RUNTIME_ROOT:-$HOME/robot-perception}"/stereo_calibration_table_checked.npz "$@"
