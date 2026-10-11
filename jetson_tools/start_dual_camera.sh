#!/usr/bin/env bash
# Local diagnostic only; apply a temporary Brio setting, test, preview, restore.
set -euo pipefail

camera_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
camera_python="$camera_dir/.venv/bin/python"
camera_helper="$camera_dir/brio_bandwidth.py"
camera_preview="$camera_dir/dual_camera_preview.py"

if [[ "${1:-}" == "--help" ]]; then
    printf 'Usage: bash %s [--test] [--brio-workaround] [preview options]\n' "${BASH_SOURCE[0]}"
    printf 'Runs a 10-second dual-stream test, then a preview. No sudo by default.\n'
    printf '%s\n' '--test exits after the test. --brio-workaround enables the legacy sudo driver workaround.'
    exit 0
fi
camera_test_only=0
camera_use_workaround=0
camera_options=()
for camera_argument in "$@"; do
    if [[ "$camera_argument" == "--test" ]]; then
        camera_test_only=1
    elif [[ "$camera_argument" == "--brio-workaround" ]]; then
        camera_use_workaround=1
    else
        camera_options+=("$camera_argument")
    fi
done
if (( EUID == 0 )); then
    printf 'Run this launcher as your normal desktop user; it calls sudo only for the driver.\n' >&2
    exit 1
fi
if [[ ! -x "$camera_python" || ! -f "$camera_preview" ]]; then
    printf 'Missing perception Python environment or dual-camera preview script.\n' >&2
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
"$camera_python" "$camera_preview" "${camera_options[@]}" --help >/dev/null
camera_launch() {
    printf '\nTesting both streams together for 10 seconds...\n'
    "$camera_python" "$camera_preview" "${camera_options[@]}" --no-display --seconds 10
    if (( ! camera_test_only )); then
        printf '\nOpening both live feeds. Press Q or Escape to close.\n'
        "$camera_python" "$camera_preview" "${camera_options[@]}"
    fi
}
if (( ! camera_use_workaround )); then
    camera_launch
    exit 0
fi
/usr/bin/python3 -I "$camera_helper" inspect

camera_applied=0
camera_cleanup() {
    local result=$?
    trap - EXIT INT TERM
    if (( camera_applied )); then
        printf '\nRestoring Brio driver defaults (sudo may request your password again).\n'
        if ! sudo /usr/bin/python3 -I "$camera_helper" restore; then
            printf 'Restore failed. Reconnect the Brio, or run:\n' >&2
            printf 'sudo /usr/bin/python3 -I %q restore\n' "$camera_helper" >&2
            result=1
        fi
    fi
    exit "$result"
}
trap camera_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf '\nApplying the temporary Brio workaround. Enter your password here if prompted.\n'
# Bash defers signal traps while waiting for sudo. Record a successful apply
# inside that deferred handler too, closing the apply/assignment cleanup gap.
camera_apply_signal() {
    local result=$?
    if (( result == 0 )); then
        camera_applied=1
    fi
    exit 130
}
trap camera_apply_signal INT TERM
if sudo /usr/bin/python3 -I "$camera_helper" apply; then
    camera_applied=1
else
    exit 1
fi
trap 'exit 130' INT
trap 'exit 143' TERM
camera_launch
