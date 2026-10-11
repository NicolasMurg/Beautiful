#!/usr/bin/env bash
# Download the local coach model and Piper voice without modifying Jetson's GPU packages.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
coach_dir="$project_dir/.local/coach"
model_id="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
piper_release="2023.11.14-2"
voice="en_US-lessac-medium"
voice_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

for dependency in curl tar python3; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
        echo "Missing $dependency. On Jetson: sudo apt install curl tar python3" >&2
        exit 1
    fi
done
if ! python3 -c "import transformers, huggingface_hub" 2>/dev/null; then
    echo "Install into the Python that runs YOLO (keep JetPack torch):" >&2
    echo "  python3 -m pip install transformers huggingface_hub" >&2
    exit 1
fi
mkdir -p "$coach_dir/voices"

# Coach language model into the standard Hugging Face cache (works offline afterwards).
python3 -c "from huggingface_hub import snapshot_download as d; print(d('$model_id'))"

# Piper text-to-speech runtime for the Jetson's 64-bit ARM CPU.
if [[ ! -x "$coach_dir/piper/piper" ]]; then
    case "$(uname -m)" in
        aarch64) archive="piper_linux_aarch64.tar.gz" ;;
        x86_64) archive="piper_linux_x86_64.tar.gz" ;;
        *) echo "No Piper release for $(uname -m); use --tts espeak." >&2; exit 1 ;;
    esac
    curl --fail --location --retry 2 \
        "https://github.com/rhasspy/piper/releases/download/$piper_release/$archive" \
        --output "$coach_dir/$archive"
    tar -xzf "$coach_dir/$archive" -C "$coach_dir"
    rm -f -- "$coach_dir/$archive"
fi
for suffix in onnx onnx.json; do
    file="$coach_dir/voices/$voice.$suffix"
    if [[ ! -s "$file" ]]; then
        curl --fail --location --retry 2 "$voice_url/$voice.$suffix" --output "$file.download"
        mv -- "$file.download" "$file"
    fi
done
echo "Ready: python3 coach.py --say \"Hello from your pool coach\""
