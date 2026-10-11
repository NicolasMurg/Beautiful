#!/usr/bin/env bash
set -euo pipefail
export TORCHDYNAMO_DISABLE=1
exec "${JETSON_RUNTIME_ROOT:-$HOME/robot-perception}"/.venv-ffs/bin/python \
  "${JETSON_RUNTIME_ROOT:-$HOME/robot-perception}"/calibrate_table_local.py \
  --manual-balls \
  --stereo-backend fast-foundation \
  --left-camera /dev/v4l/by-id/usb-Jieli_Technology_USB_PHY_2.0-video-index0 \
  --right-camera /dev/v4l/by-id/usb-046d_1080P_Pro_Stream_C550F4FF-video-index0 \
  --stereo-calibration "${JETSON_RUNTIME_ROOT:-$HOME/robot-perception}"/stereo_calibration_table.npz \
  --baseline-inches 4.25 \
  --table-length-cm 223.52 --table-width-cm 111.76 \
  --table-long-axis top-to-bottom \
  --ball-diameter-cm 5.715 --pocket-width-cm 11.5 "$@"
