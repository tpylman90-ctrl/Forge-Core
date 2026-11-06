#!/usr/bin/env bash
set -euo pipefail

PIN="3f8af15a76ac8675812e856da45454bafc90f12c"
DEST="vendor/llama.cpp"
mkdir -p vendor
if [[ ! -d "$DEST/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "$DEST"
fi
git -C "$DEST" fetch --depth=1 origin "$PIN"
git -C "$DEST" checkout --detach "$PIN"
cp scripts/llama-module.gradle "$DEST/examples/llama.android/lib/build.gradle"
sed -i 's/DEFAULT_CONTEXT_SIZE    = 8192/DEFAULT_CONTEXT_SIZE    = 2048/' \
  "$DEST/examples/llama.android/lib/src/main/cpp/ai_chat.cpp"
python3 scripts/patch-llama-android.py
echo "Pinned llama.cpp Android source ready at $PIN"
