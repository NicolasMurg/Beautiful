#!/usr/bin/env bash
# Build an isolated CPU Whisper runtime without modifying Jetson's GPU packages.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$project_dir/.local/whisper.cpp"
model_dir="$project_dir/models/whisper"
model_name="${1:-tiny.en}"
version="v1.7.6"

case "$model_name" in
    tiny.en|base.en|small.en) ;;
    *) echo "Choose tiny.en, base.en, or small.en." >&2; exit 1 ;;
esac
for dependency in git cmake c++ curl arecord; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
        echo "Missing $dependency. On Jetson: sudo apt install git cmake build-essential curl alsa-utils" >&2
        exit 1
    fi
done
mkdir -p "$project_dir/.local" "$model_dir"
if [[ ! -d "$source_dir" ]]; then
    git clone --depth 1 --branch "$version" https://github.com/ggml-org/whisper.cpp.git "$source_dir"
elif [[ "$(git -C "$source_dir" describe --tags --exact-match 2>/dev/null || true)" != "$version" ]]; then
    echo "$source_dir already exists with a different version; use --whisper-bin for that runtime." >&2
    exit 1
fi
cmake -S "$source_dir" -B "$source_dir/build" -DGGML_CUDA=OFF \
    -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF
cmake --build "$source_dir/build" --target whisper-cli --parallel 2
model_file="$model_dir/ggml-$model_name.bin"
if [[ ! -s "$model_file" ]]; then
    curl --fail --location --retry 2 \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$model_name.bin" \
        --output "$model_file.download"
    mv -- "$model_file.download" "$model_file"
fi
echo "Ready: python3 voice_to_text.py --whisper-model models/whisper/ggml-$model_name.bin"
