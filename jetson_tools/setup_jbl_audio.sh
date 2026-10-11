#!/usr/bin/env bash
set -euo pipefail
override=/etc/systemd/system/bluetooth.service.d/zz-jetson-speaker-audio.conf
echo 'Enable Bluetooth speaker audio (A2DP/AVRCP) on this Jetson.'
echo 'Power on JBL Flip Essential 2. Disconnect it from your phone if necessary.'
if [[ -e "$override" ]]; then
    if ! cmp -s "$override" <(printf '[Service]\nExecStart=\nExecStart=/usr/libexec/bluetooth/bluetoothd --noplugin=sap\n'); then
        echo "Existing custom override differs: $override. Stopping to preserve it." >&2
        exit 1
    fi
else
    sudo install -d -m 0755 /etc/systemd/system/bluetooth.service.d
    printf '[Service]\nExecStart=\nExecStart=/usr/libexec/bluetooth/bluetoothd --noplugin=sap\n' | sudo tee "$override" >/dev/null
    sudo chmod 0644 "$override"
fi
sudo systemctl daemon-reload
sudo systemctl restart bluetooth
sleep 2
echo 'Bluetooth audio enabled. Starting the seven-second JBL tone test.'
python3 "${JETSON_RUNTIME_ROOT:-$HOME/robot-perception}"/jbl_speaker_test.py --set-default
