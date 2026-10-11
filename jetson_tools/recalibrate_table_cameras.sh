#!/usr/bin/env bash
set -euo pipefail
exec "${JETSON_RUNTIME_ROOT:-$HOME/robot-perception}"/.venv/bin/python \
  "${JETSON_RUNTIME_ROOT:-$HOME/robot-perception}"/calibrate_stereo_local.py \
  --left-camera /dev/v4l/by-id/usb-Jieli_Technology_USB_PHY_2.0-video-index0 \
  --right-camera /dev/v4l/by-id/usb-046d_1080P_Pro_Stream_C550F4FF-video-index0 \
  --square-mm 25.4 --baseline-inches 4.25 \
  --output "${JETSON_RUNTIME_ROOT:-$HOME/robot-perception}"/stereo_calibration_table.npz "$@"
