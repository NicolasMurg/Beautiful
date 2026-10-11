#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' 'Restoring the previous 25W mode. Enter your Jetson password below.'
sudo /usr/sbin/nvpmodel -m 1
/usr/sbin/nvpmodel -q
printf '%s\n' 'Starting the previous uncapped YOLOE + Fast-FoundationStereo feed (1000ms reading age).'
exec bash "${JETSON_RUNTIME_ROOT:-$HOME/robot-perception}"/start_can_stereo.sh
